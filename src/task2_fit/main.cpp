// ===========================================================================
//  任务 2：合成旋转视频的参数拟合
//
//  输入：resources/task_2.mp4（960×720，60 FPS，1440 帧，相机固定）
//  模型：角速度 ω(t) = b + A·sin(Ωt + φ)
//        积分得角度 θ(t) = θ0 + b·t − (A/Ω)·[cos(Ωt + φ) − cos φ]，θ0 = θ(0)
//        已知 A > 0、b > A、Ω > 0，目标持续沿同一方向转动。
//
//  步骤：
//    1. 逐帧做 HSV 阈值分割识别青色目标，取最大连通域质心 (x_i, y_i)；
//       同时统计白色圆点位置，用来核对已知旋转中心 (480, 360)。
//    2. 求 θ_wrapped,i = atan2(c_y − y_i, x_i − c_x)，再沿时间展开为连续角度。
//    3. 参数估计（两步）：
//       a) 固定 Ω 时模型对 (c0, b, c2, c3) 是线性的，用 Eigen 列主元 QR 解线性
//          最小二乘；对 Ω 做一维网格搜索 + 逐轮细化，得到初值。
//       b) 用 Ceres 自动求导对 (θ0, b, A, Ω, φ) 做非线性精修，并施加 A > 0、
//          Ω > 0 的下界约束。
//    4. 输出标注视频、拟合对比图、角速度曲线、残差图，以及误差指标和求解状态。
// ===========================================================================

#include <opencv2/opencv.hpp>

#include <Eigen/Dense>
#include <ceres/ceres.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "plot_utils.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

// ------------------------- 目标检测参数 -------------------------
// 青色目标实测 H≈90、S≈200、V≈210，面积稳定在 531~549 像素
constexpr int kCyanHMin = 80, kCyanHMax = 105;
constexpr int kCyanSMin = 80, kCyanVMin = 80;
constexpr double kMinTargetArea = 100.0;
// 白色旋转中心：高亮度、低饱和度
constexpr int kWhiteSMax = 40, kWhiteVMin = 200;
constexpr double kMinCenterArea = 20.0;

// ------------------------- 拟合参数 -------------------------
constexpr double kOmegaMin = 0.05, kOmegaMax = 8.0;  // Ω 搜索范围
constexpr int kCoarseSteps = 2000;                   // 粗搜索步数
constexpr int kRefineRounds = 3;                     // 逐轮细化次数
constexpr int kRefineSteps = 200;                    // 每轮细化步数

// 一帧数据
struct Sample {
    int frame = 0;
    double t = 0.0;             // 相对第 0 帧的时间，单位 s
    double x = 0.0, y = 0.0;    // 目标质心 (px)
    double area = 0.0;          // 目标面积 (px²)
    double thetaWrapped = 0.0;  // atan2 得到的 [-π, π) 角度
    double theta = 0.0;         // 展开后的连续角度 (rad)
    bool valid = false;
};

// 待估参数
struct Model {
    double theta0 = 0.0;  // θ(0)，rad
    double b = 0.0;       // 平均角速度，rad/s
    double A = 0.0;       // 角速度振幅，rad/s
    double Omega = 0.0;   // 角速度变化的频率参数，rad/s
    double phi = 0.0;     // 相位，rad
};

double wrapToPi(double a) {
    a = std::fmod(a + kPi, kTwoPi);
    if (a < 0.0) a += kTwoPi;
    return a - kPi;
}

double thetaModel(double t, const Model& m) {
    return m.theta0 + m.b * t -
           (m.A / m.Omega) * (std::cos(m.Omega * t + m.phi) - std::cos(m.phi));
}

double omegaModel(double t, const Model& m) {
    return m.b + m.A * std::sin(m.Omega * t + m.phi);
}

// Ceres 残差块：r = θ_obs − θ_model，参数块为 {θ0, b, A, Ω, φ}
struct AngleResidual {
    AngleResidual(double t, double theta) : t_(t), theta_(theta) {}

    template <typename T>
    bool operator()(const T* const p, T* residual) const {
        using std::cos;  // 让自动求导类型也能取到正确的重载
        using std::sin;
        const T model = p[0] + p[1] * T(t_) -
                        (p[2] / p[3]) * (cos(p[3] * T(t_) + p[4]) - cos(p[4]));
        residual[0] = T(theta_) - model;
        return true;
    }

private:
    double t_, theta_;
};

// 取掩膜中面积最大的连通域质心
bool largestBlobCentroid(const cv::Mat& mask, double minArea, cv::Point2d& centroid,
                         double& area) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    int best = -1;
    double bestArea = 0.0;
    for (size_t i = 0; i < contours.size(); ++i) {
        const double a = cv::contourArea(contours[i]);
        if (a > bestArea) {
            bestArea = a;
            best = static_cast<int>(i);
        }
    }
    if (best < 0 || bestArea < minArea) return false;
    const cv::Moments m = cv::moments(contours[best]);
    if (m.m00 <= 0.0) return false;
    centroid = cv::Point2d(m.m10 / m.m00, m.m01 / m.m00);
    area = bestArea;
    return true;
}

// 给定 Ω 时模型对 (c0, b, c2, c3) 线性，用 Eigen 列主元 QR 求最小二乘解，
// 返回残差平方和。θ(t) = c0 + b·t + c2·cos(Ωt) + c3·sin(Ωt)
double solveLinearForOmega(const std::vector<Sample>& s, double omega,
                           Eigen::Vector4d& coef) {
    const int n = static_cast<int>(s.size());
    Eigen::MatrixXd design(n, 4);
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        const double t = s[i].t;
        design(i, 0) = 1.0;
        design(i, 1) = t;
        design(i, 2) = std::cos(omega * t);
        design(i, 3) = std::sin(omega * t);
        y(i) = s[i].theta;
    }
    coef = design.colPivHouseholderQr().solve(y);
    return (design * coef - y).squaredNorm();
}

// 线性系数换算成物理参数：c0 = θ0 − c2，c2 = −(A/Ω)cosφ，c3 = (A/Ω)sinφ
Model modelFromLinear(const Eigen::Vector4d& c, double omega) {
    Model m;
    m.Omega = omega;
    m.b = c[1];
    m.A = omega * std::hypot(c[2], c[3]);
    m.phi = std::atan2(c[3], -c[2]);
    m.theta0 = c[0] + c[2];
    return m;
}

// 画半透明底 + 文字，保证叠加信息在任意背景上都清晰
void drawHud(cv::Mat& img, const std::vector<std::string>& lines, cv::Point org,
             const std::vector<cv::Scalar>& colors) {
    const double scale = 0.48;
    const int lineH = 27;
    int maxw = 0;
    for (const auto& s : lines) {
        int baseline = 0;
        maxw = std::max(
                   maxw,
                   cv::getTextSize(s, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline).width);
    }
    const int panelW = std::min(maxw + 28, img.cols - org.x);
    const int panelH = std::min(lineH * static_cast<int>(lines.size()) + 18, img.rows - org.y);
    cv::Mat roi = img(cv::Rect(org.x, org.y, panelW, panelH));
    roi.convertTo(roi, -1, 0.35);  // 压暗背景，便于阅读
    for (size_t i = 0; i < lines.size(); ++i) {
        cv::putText(img, lines[i],
                    cv::Point(org.x + 14, org.y + 32 + static_cast<int>(i) * lineH),
                    cv::FONT_HERSHEY_SIMPLEX, scale, colors[i], 1, cv::LINE_AA);
    }
}

std::string fmt(const char* f, double v) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), f, v);
    return std::string(buf);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string inPath = argc > 1 ? argv[1] : "resources/task_2.mp4";
    const std::string outDir = argc > 2 ? argv[2] : "result/task2_fit";
    std::filesystem::create_directories(outDir);

    cv::VideoCapture cap(inPath);
    if (!cap.isOpened()) {
        std::cerr << "无法打开视频: " << inPath << "\n";
        return 1;
    }
    const double fps = cap.get(cv::CAP_PROP_FPS);
    const int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    const double dt = 1.0 / fps;
    const double duration = totalFrames / fps;
    std::cout << "视频: " << width << "x" << height << ", " << fps << " FPS, " << totalFrames
              << " 帧, " << duration << " s\n";

    // 已知旋转中心（讲义给定），另用白点检测做核对
    const cv::Point2d center(480.0, 360.0);

    // ---------------- 第一遍：逐帧检测目标 ----------------
    std::vector<Sample> samples;
    samples.reserve(totalFrames);
    std::vector<cv::Point2d> whiteCenters;
    std::vector<double> whiteAreas;
    cv::Mat frame, hsv, cyanMask, whiteMask;
    int idx = 0;
    while (cap.read(frame)) {
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        Sample s;
        s.frame = idx;
        s.t = idx * dt;
        cv::inRange(hsv, cv::Scalar(kCyanHMin, kCyanSMin, kCyanVMin),
                    cv::Scalar(kCyanHMax, 255, 255), cyanMask);
        cv::Point2d c;
        double area = 0.0;
        if (largestBlobCentroid(cyanMask, kMinTargetArea, c, area)) {
            s.valid = true;
            s.x = c.x;
            s.y = c.y;
            s.area = area;
        }
        samples.push_back(s);

        cv::inRange(hsv, cv::Scalar(0, 0, kWhiteVMin), cv::Scalar(179, kWhiteSMax, 255),
                    whiteMask);
        if (largestBlobCentroid(whiteMask, kMinCenterArea, c, area)) {
            whiteCenters.push_back(c);
            whiteAreas.push_back(area);
        }
        ++idx;
    }
    cap.release();

    if (samples.empty()) {
        std::cerr << "视频中没有读到任何帧\n";
        return 1;
    }

    // ---------------- 角度计算与展开 ----------------
    // 只有相邻帧角差远小于 π 时才可能展开错误；实测每帧转过约 0.02~0.04 rad。
    std::vector<Sample> valid;
    valid.reserve(samples.size());
    double prevWrapped = 0.0;
    double unwrapped = 0.0;
    bool hasPrev = false;
    for (auto& s : samples) {
        if (!s.valid) continue;
        s.thetaWrapped = std::atan2(center.y - s.y, s.x - center.x);
        if (!hasPrev) {
            unwrapped = s.thetaWrapped;
        } else {
            double d = s.thetaWrapped - prevWrapped;
            while (d > kPi) d -= kTwoPi;
            while (d < -kPi) d += kTwoPi;
            unwrapped += d;
        }
        s.theta = unwrapped;
        prevWrapped = s.thetaWrapped;
        hasPrev = true;
        valid.push_back(s);
    }
    if (valid.size() < 10) {
        std::cerr << "有效目标过少: " << valid.size() << "\n";
        return 1;
    }

    const int firstFrame = valid.front().frame;
    const int lastFrame = valid.back().frame;
    const double firstT = valid.front().t;
    const double lastT = valid.back().t;

    // ---------------- 第 1 步：Eigen 线性最小二乘 + Ω 网格搜索 ----------------
    Eigen::Vector4d coef;
    double bestSse = std::numeric_limits<double>::max();
    double bestOmega = kOmegaMin;
    const double coarseStep = (kOmegaMax - kOmegaMin) / kCoarseSteps;
    for (int i = 0; i <= kCoarseSteps; ++i) {
        const double omega = kOmegaMin + i * coarseStep;
        const double sse = solveLinearForOmega(valid, omega, coef);
        if (sse < bestSse) {
            bestSse = sse;
            bestOmega = omega;
        }
    }
    double half = coarseStep;
    for (int r = 0; r < kRefineRounds; ++r) {
        const double lo = std::max(1e-3, bestOmega - half);
        const double hi = bestOmega + half;
        for (int i = 0; i <= kRefineSteps; ++i) {
            const double omega = lo + (hi - lo) * i / kRefineSteps;
            const double sse = solveLinearForOmega(valid, omega, coef);
            if (sse < bestSse) {
                bestSse = sse;
                bestOmega = omega;
            }
        }
        half = (hi - lo) / kRefineSteps;
    }
    solveLinearForOmega(valid, bestOmega, coef);
    const Model init = modelFromLinear(coef, bestOmega);

    // ---------------- 第 2 步：Ceres 非线性精修 ----------------
    Model fit = init;
    double params[5] = {init.theta0, init.b, init.A, init.Omega, init.phi};
    ceres::Problem problem;
    for (const auto& s : valid) {
        auto* cost = new ceres::AutoDiffCostFunction<AngleResidual, 1, 5>(
            new AngleResidual(s.t, s.theta));
        problem.AddResidualBlock(cost, nullptr, params);
    }
    problem.SetParameterLowerBound(params, 2, 1e-6);  // A > 0
    problem.SetParameterLowerBound(params, 3, 1e-3);  // Ω > 0
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 200;
    options.function_tolerance = 1e-14;
    options.gradient_tolerance = 1e-14;
    options.parameter_tolerance = 1e-14;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    fit.theta0 = params[0];
    fit.b = params[1];
    fit.A = params[2];
    fit.Omega = params[3];
    fit.phi = wrapToPi(params[4]);

    // ---------------- 误差指标 ----------------
    double sseTheta = 0.0, maxAbsTheta = 0.0;
    for (const auto& s : valid) {
        const double r = s.theta - thetaModel(s.t, fit);
        sseTheta += r * r;
        maxAbsTheta = std::max(maxAbsTheta, std::fabs(r));
    }
    const double rmseTheta = std::sqrt(sseTheta / valid.size());

    // 角速度观测值用中心差分得到（只取帧号连续的三点）
    std::vector<cv::Point2d> omegaObs;
    double sseOmega = 0.0, maxAbsOmega = 0.0;
    int omegaCount = 0;
    for (size_t i = 1; i + 1 < valid.size(); ++i) {
        if (valid[i].frame - valid[i - 1].frame != 1 ||
                valid[i + 1].frame - valid[i].frame != 1)
            continue;
        const double w = (valid[i + 1].theta - valid[i - 1].theta) /
                         (valid[i + 1].t - valid[i - 1].t);
        const double wm = omegaModel(valid[i].t, fit);
        omegaObs.emplace_back(valid[i].t, w);
        sseOmega += (w - wm) * (w - wm);
        maxAbsOmega = std::max(maxAbsOmega, std::fabs(w - wm));
        ++omegaCount;
    }
    const double rmseOmega = omegaCount > 0 ? std::sqrt(sseOmega / omegaCount) : 0.0;

    // 旋转半径统计（核对 220 px）
    double rSum = 0.0, rMin = 1e9, rMax = 0.0;
    for (const auto& s : valid) {
        const double r = std::hypot(s.x - center.x, center.y - s.y);
        rSum += r;
        rMin = std::min(rMin, r);
        rMax = std::max(rMax, r);
    }
    const double rMean = rSum / valid.size();

    cv::Point2d whiteMean(0, 0);
    double whiteMeanArea = 0.0;
    if (!whiteCenters.empty()) {
        for (const auto& p : whiteCenters) whiteMean += p;
        whiteMean.x /= whiteCenters.size();
        whiteMean.y /= whiteCenters.size();
        for (double a : whiteAreas) whiteMeanArea += a;
        whiteMeanArea /= whiteAreas.size();
    }

    // ---------------- 控制台 / 文本报告 ----------------
    std::ostringstream report;
    report << std::fixed << std::setprecision(6);
    report << "=========== 任务 2：合成旋转视频参数拟合 ===========\n";
    report << "输入视频      : " << inPath << "\n";
    report << "分辨率/帧率   : " << width << "x" << height << " / " << fps << " FPS\n";
    report << "总帧数        : " << totalFrames << "（时长 " << duration << " s）\n";
    report << "有效观测帧    : " << valid.size() << "，帧范围 [" << firstFrame << ", "
           << lastFrame << "]，时间 [" << firstT << ", " << lastT << "] s\n";
    report << "已知旋转中心  : (" << center.x << ", " << center.y << ") px\n";
    if (!whiteCenters.empty()) {
        report << "白点检测中心  : (" << whiteMean.x << ", " << whiteMean.y << ") px，面积 "
               << whiteMeanArea << " px²，检出 " << whiteCenters.size() << " 帧\n";
    }
    report << "旋转半径      : 均值 " << rMean << " px（min " << rMin << " / max " << rMax
           << "，理论 220）\n";
    report << "\n--- 线性最小二乘初值（Eigen，Ω 网格搜索）---\n";
    report << "A      = " << init.A << " rad/s\n";
    report << "b      = " << init.b << " rad/s\n";
    report << "Omega  = " << init.Omega << " rad/s\n";
    report << "phi    = " << init.phi << " rad\n";
    report << "theta0 = " << init.theta0 << " rad\n";
    report << "\n--- Ceres 精修结果 ---\n";
    report << "A      = " << fit.A << " rad/s\n";
    report << "b      = " << fit.b << " rad/s\n";
    report << "Omega  = " << fit.Omega << " rad/s  (周期 T = " << kTwoPi / fit.Omega
           << " s)\n";
    report << "phi    = " << fit.phi << " rad\n";
    report << "theta0 = " << fit.theta0 << " rad  (" << fit.theta0 * 180.0 / kPi << " deg)\n";
    report << "b - A  = " << fit.b - fit.A << " rad/s (需 > 0)\n";
    report << "求解状态      : " << summary.BriefReport() << "\n";
    report << "可用解        : " << (summary.IsSolutionUsable() ? "yes" : "no") << "\n";
    report << "\n--- 误差指标 ---\n";
    report << std::scientific << std::setprecision(6);
    report << "RMSE(theta)   = " << rmseTheta << " rad\n";
    report << "max|theta err|= " << maxAbsTheta << " rad\n";
    report << "RMSE(omega)   = " << rmseOmega << " rad/s（中心差分观测，样本数 " << omegaCount
           << "，帧范围 [" << firstFrame + 1 << ", " << lastFrame - 1 << "]）\n";
    report << "max|omega err|= " << maxAbsOmega << " rad/s\n";
    std::cout << "\n" << report.str();
    std::ofstream(outDir + "/fit_report.txt") << report.str();

    // ---------------- 结果图 ----------------
    const cv::Scalar kObsColor(200, 120, 0);  // 观测：蓝色
    const cv::Scalar kFitColor(0, 0, 220);    // 拟合：红色

    std::vector<cv::Point2d> thetaObsPts;
    thetaObsPts.reserve(valid.size());
    for (const auto& s : valid) thetaObsPts.emplace_back(s.t, s.theta);

    std::vector<cv::Point2d> thetaFitPts, omegaModelPts;
    const int denseN = 2000;
    for (int i = 0; i <= denseN; ++i) {
        const double t = firstT + (lastT - firstT) * i / denseN;
        thetaFitPts.emplace_back(t, thetaModel(t, fit));
        omegaModelPts.emplace_back(t, omegaModel(t, fit));
    }

    const auto yRange = [](const std::vector<cv::Point2d>& a,
                           const std::vector<cv::Point2d>& b, double frac,
    double floorPad) {
        double lo = 1e18, hi = -1e18;
        for (const auto& p : a) {
            lo = std::min(lo, p.y);
            hi = std::max(hi, p.y);
        }
        for (const auto& p : b) {
            lo = std::min(lo, p.y);
            hi = std::max(hi, p.y);
        }
        const double pad = std::max(floorPad, (hi - lo) * frac);
        return std::make_pair(lo - pad, hi + pad);
    };

    // 1) 观测点 + 拟合曲线（角度）
    {
        const auto yr = yRange(thetaObsPts, thetaFitPts, 0.06, 0.05);
        plot::Figure fig(1400, 800, "Task 2 - rotation angle: observation vs fit",
                         "time t (s)", "theta (rad)");
        fig.setRange(firstT, lastT, yr.first, yr.second);
        fig.drawAxes();
        fig.drawPoints(thetaObsPts, kObsColor, 3, 6);
        fig.drawCurve(thetaFitPts, kFitColor, 3);
        fig.drawLegend({{"observation (unwrapped)", kObsColor}, {"fitted curve", kFitColor}});
        fig.save(outDir + "/fit_comparison.png");
    }

    // 2) 角速度曲线
    {
        const auto yr = yRange(omegaObs, omegaModelPts, 0.10, 0.02);
        plot::Figure fig(1400, 800, "Task 2 - angular velocity: observation vs model",
                         "time t (s)", "omega (rad/s)");
        fig.setRange(firstT, lastT, yr.first, yr.second);
        fig.drawAxes();
        fig.drawPoints(omegaObs, kObsColor, 2, 6);
        fig.drawCurve(omegaModelPts, kFitColor, 3);
        fig.drawLegend({{"central difference", kObsColor}, {"fitted model", kFitColor}});
        fig.save(outDir + "/angular_velocity.png");
    }

    // 3) 残差图
    {
        std::vector<cv::Point2d> residualPts;
        residualPts.reserve(valid.size());
        for (const auto& s : valid) {
            residualPts.emplace_back(s.t, s.theta - thetaModel(s.t, fit));
        }
        const auto yr = yRange(residualPts, {}, 0.15, 1e-6);
        plot::Figure fig(1400, 800, "Task 2 - residual: theta_obs - theta_fit", "time t (s)",
                         "residual (rad)");
        fig.setRange(firstT, lastT, yr.first, yr.second);
        fig.drawAxes();
        fig.drawHorizontalLine(0.0, cv::Scalar(120, 120, 120), 1);
        fig.drawPoints(residualPts, kObsColor, 2, 6);
        fig.drawLegend({{"residual", kObsColor}});
        fig.save(outDir + "/residuals.png");
    }

    // ---------------- 标注视频 ----------------
    // 小图背景只画一次，循环内只做复制 + 画游标
    plot::Figure miniFig(560, 210, "", "t (s)", "theta (rad)");
    miniFig.setPadding(80, 20, 20, 55);
    {
        const auto yr = yRange(thetaObsPts, thetaFitPts, 0.06, 0.05);
        miniFig.setRange(firstT, lastT, yr.first, yr.second);
        miniFig.drawAxes();
        miniFig.drawPoints(thetaObsPts, kObsColor, 2, 8);
        miniFig.drawCurve(thetaFitPts, kFitColor, 2);
    }
    const cv::Mat miniBg = miniFig.canvas().clone();

    cv::VideoCapture cap2(inPath);
    cv::VideoWriter writer(outDir + "/tracking_overlay.mp4",
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                           cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "无法创建输出视频，请检查 OpenCV 是否带 FFMPEG 后端\n";
        return 1;
    }

    const cv::Scalar kWhite(255, 255, 255);
    const cv::Scalar kGreen(0, 255, 0);
    const cv::Scalar kYellow(0, 255, 255);
    const cv::Scalar kRed(0, 0, 255);
    const cv::Point centerPx(cvRound(center.x), cvRound(center.y));

    idx = 0;
    int writeCount = 0;
    while (idx < static_cast<int>(samples.size()) && cap2.read(frame)) {
        cv::Mat vis = frame.clone();
        const Sample& s = samples[idx];

        // 旋转轨迹参考圆 + 中心标记
        cv::circle(vis, centerPx, 220, cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
        cv::drawMarker(vis, centerPx, kWhite, cv::MARKER_CROSS, 28, 2, cv::LINE_AA);
        cv::circle(vis, centerPx, 5, kWhite, 1, cv::LINE_AA);

        double radius = 0.0;
        if (s.valid) radius = std::hypot(s.x - center.x, center.y - s.y);

        std::vector<std::string> lines;
        std::vector<cv::Scalar> colors;
        lines.push_back("task 2  synthetic rotation fitting");
        colors.push_back(kWhite);
        lines.push_back("frame " + std::to_string(idx) + " / " + std::to_string(totalFrames) +
                        "    t = " + fmt("%.3f", s.t) + " s");
        colors.push_back(kWhite);
        if (s.valid) {
            lines.push_back("centroid (" + fmt("%.1f", s.x) + ", " + fmt("%.1f", s.y) +
                            ") px    area " + fmt("%.0f", s.area));
            colors.push_back(kGreen);
            lines.push_back("theta " + fmt("%.2f", s.thetaWrapped * 180.0 / kPi) +
                            " deg   unwrapped " + fmt("%.2f", s.theta * 180.0 / kPi) + " deg");
            colors.push_back(kGreen);
            lines.push_back("omega " + fmt("%.3f", omegaModel(s.t, fit)) + " rad/s (model)" +
                            "    radius " + fmt("%.1f", radius) + " px");
            colors.push_back(kYellow);
            lines.push_back("status: DETECTED");
            colors.push_back(kGreen);
        } else {
            lines.push_back("target not found");
            colors.push_back(kRed);
            lines.push_back("status: LOST");
            colors.push_back(kRed);
        }
        drawHud(vis, lines, cv::Point(12, 12), colors);

        // 左下角：θ(t) 观测与拟合 + 当前帧游标
        cv::Mat mini = miniBg.clone();
        if (s.valid) {
            const cv::Point mp = miniFig.toPixel(s.t, s.theta);
            cv::line(mini, cv::Point(mp.x, 20), cv::Point(mp.x, mini.rows - 55),
                     cv::Scalar(180, 180, 180), 1, cv::LINE_AA);
            cv::circle(mini, mp, 6, kRed, cv::FILLED, cv::LINE_AA);
        }
        const int px = 12, py = height - mini.rows - 12;
        if (px + mini.cols <= width && py >= 0) {
            const cv::Rect roi(px, py, mini.cols, mini.rows);
            cv::addWeighted(vis(roi), 0.25, mini, 0.75, 0.0, vis(roi));
        }

        // 目标标记最后绘制，保证不被左下角小图或顶部 HUD 遮挡
        if (s.valid) {
            const cv::Point tp(cvRound(s.x), cvRound(s.y));
            cv::line(vis, centerPx, tp, kYellow, 1, cv::LINE_AA);
            cv::circle(vis, tp, 18, kGreen, 2, cv::LINE_AA);
            cv::circle(vis, tp, 3, kGreen, cv::FILLED, cv::LINE_AA);
        }

        writer.write(vis);
        ++writeCount;
        ++idx;
    }
    writer.release();
    cap2.release();

    std::cout << "\n输出目录: " << outDir << "\n"
              << "  tracking_overlay.mp4  (" << writeCount << " 帧)\n"
              << "  fit_comparison.png / angular_velocity.png / residuals.png\n"
              << "  fit_report.txt\n";
    return 0;
}
