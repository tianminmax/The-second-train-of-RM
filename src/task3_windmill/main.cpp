// ===========================================================================
//  任务 3：真实能量机关视频的识别与稳定跟踪
//
//  输入：resources/task_3.mp4（小能量机关，同时最多亮 1 个目标）
//        resources/task_4.mp4（大能量机关，同时最多亮 2 个目标）
//
//  与原视频一致：1440×1080、30 FPS。真实录像没有可靠的逐帧采集时间，
//  本任务只做视觉识别与稳定跟踪，不做时间拟合。
//
//  处理流程（逐帧）：
//    1. 暖色掩膜（机关本体 + 灯板）→ 加权质心作为「R 标中心」（机关中心），
//       并做指数平滑抑制抖动；掩膜过少时用上一帧位置外推。
//    2. 高亮灯珠掩膜 → 形态学合并同一扇叶上的灯珠 → 每个连通域是一个候选目标，
//       用 minEnclosingCircle 拟合出「扇叶圆」（圆心 + 半径）。
//    3. 身份关联：把候选目标与当前锁定的目标按距离关联（带速度和关联门限），
//       只有关联成功的候选才继承原 ID，避免直接用轮廓列表顺序当 ID。
//    4. 状态机：IDLE → ACQUIRE → DETECTED → LOST（容忍若干帧）→ 重新选择。
//
//  输出：每个视频一个覆盖完整时段的 recognition_overlay.mp4，
//        以及可选的 binary_process.mp4（中间二值化过程）。
// ===========================================================================

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

// ------------------------- 检测参数 -------------------------
// 机关本体与灯板：暖色（红/橙/黄），亮度较低也算
constexpr int kWarmHMax = 45;      // H < 45 或 H >= 165 视为暖色
constexpr int kWarmHMinHigh = 165;
constexpr int kWarmSMin = 90;
constexpr int kWarmVMin = 25;

// 灯珠（点亮的目标）：更高饱和度与亮度
constexpr int kLedHMax = 35;
constexpr int kLedSMin = 140;
constexpr int kLedVMin = 150;

constexpr int kMergeRadius = 20;         // 合并同一扇叶灯珠的闭运算半径
constexpr double kMinClusterArea = 500;  // 候选目标最小面积（滤掉零星反光点）

// ------------------------- 跟踪参数 -------------------------
constexpr double kAssocGate = 110.0;     // 关联门限（像素），再按半径放宽并封顶
constexpr double kAssocGateMax = 200.0;  // 门限上限，避免半径很大时误关联
constexpr double kRadiusEma = 0.35;      // 扇叶圆半径平滑系数，抑制轮廓跳变
constexpr int kLostTolerance = 30;       // 连续丢失多少帧后允许重选（30 帧 = 1 s）
constexpr double kCenterEma = 0.35;      // R 标中心平滑系数
constexpr double kVelocityEma = 0.5;     // 速度平滑系数

struct Candidate {          // 一帧中的一个候选目标（扇叶）
    cv::Point2d centre;     // 扇叶圆中心
    double radius = 0.0;    // 扇叶圆半径
    double area = 0.0;
    int frame = 0;
};

struct FrameResult {
    int frame = 0;
    bool centreValid = false;   // 本帧是否检测到机关中心
    cv::Point2d centre;         // R 标中心
    std::vector<Candidate> candidates;
    // 跟踪状态
    int selected = -1;          // 当前选择的目标在 candidates 中的下标，-1 表示无
    int targetId = -1;          // 目标 ID
    bool detected = false;      // 本帧是否有效观测
    bool lost = false;          // 是否处于丢失状态
    cv::Point2d targetCircleCentre;  // 画面上显示的扇叶圆中心（丢失时为预测值）
    double targetRadius = 0.0;
};

// 暖色掩膜：机关本体 + 灯板
void warmMask(const cv::Mat& hsv, cv::Mat& mask) {
    cv::Mat lo, hi;
    cv::inRange(hsv, cv::Scalar(0, kWarmSMin, kWarmVMin),
                cv::Scalar(kWarmHMax, 255, 255), lo);
    cv::inRange(hsv, cv::Scalar(kWarmHMinHigh, kWarmSMin, kWarmVMin),
                cv::Scalar(179, 255, 255), hi);
    cv::bitwise_or(lo, hi, mask);
}

// 灯珠掩膜：点亮的目标
void ledMask(const cv::Mat& hsv, cv::Mat& mask) {
    cv::Mat lo, hi;
    cv::inRange(hsv, cv::Scalar(0, kLedSMin, kLedVMin), cv::Scalar(kLedHMax, 255, 255), lo);
    cv::inRange(hsv, cv::Scalar(kWarmHMinHigh, kLedSMin, kLedVMin),
                cv::Scalar(179, 255, 255), hi);
    cv::bitwise_or(lo, hi, mask);
}

// 用亮度加权质心估计机关中心：比二值质心更稳，且用 moments 实现，速度快。
// weighted 需预分配为同尺寸的 8 位图，这里复用以免每帧都申请大块内存。
cv::Point2d weightedCentroid(cv::Mat& weighted, const cv::Mat& vChannel, const cv::Mat& mask,
                             double minPixels, bool& ok) {
    ok = false;
    if (cv::countNonZero(mask) < minPixels) return {0, 0};
    weighted.setTo(0);
    vChannel.copyTo(weighted, mask);          // 只在掩膜内保留亮度，其余置 0
    const cv::Moments m = cv::moments(weighted, false);   // 亮度加权的一阶矩
    if (m.m00 <= 0.0) return {0, 0};
    ok = true;
    return {m.m10 / m.m00, m.m01 / m.m00};
}

std::string fmt(const char* f, double v) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), f, v);
    return std::string(buf);
}

// 半透明底 + 多行文字
void drawHud(cv::Mat& img, const std::vector<std::string>& lines,
             const std::vector<cv::Scalar>& colors, cv::Point org) {
    const double scale = 0.52;
    const int lineH = 29;
    int maxw = 0;
    for (const auto& s : lines) {
        int baseline = 0;
        maxw = std::max(
            maxw, cv::getTextSize(s, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline).width);
    }
    const int w = std::min(maxw + 28, img.cols - org.x);
    const int h = std::min(lineH * static_cast<int>(lines.size()) + 18, img.rows - org.y);
    cv::Mat roi = img(cv::Rect(org.x, org.y, w, h));
    roi.convertTo(roi, -1, 0.35);
    for (size_t i = 0; i < lines.size(); ++i) {
        cv::putText(img, lines[i], cv::Point(org.x + 14, org.y + 33 + static_cast<int>(i) * lineH),
                    cv::FONT_HERSHEY_SIMPLEX, scale, colors[i], 1, cv::LINE_AA);
    }
}

// 处理单个视频
int processVideo(const std::string& inPath, const std::string& outDir, bool writeBinary,
                 int maxFrames = 0) {
    std::filesystem::create_directories(outDir);
    cv::VideoCapture cap(inPath);
    if (!cap.isOpened()) {
        std::cerr << "无法打开视频: " << inPath << "\n";
        return 1;
    }
    const double fps = cap.get(cv::CAP_PROP_FPS);
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    if (maxFrames > 0) totalFrames = std::min(totalFrames, maxFrames);
    const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "\n视频 " << inPath << ": " << width << "x" << height << ", " << fps
              << " FPS, " << totalFrames << " 帧\n";

    const bool debug = std::getenv("T3_DEBUG") != nullptr;
    // 闭运算把同一扇叶上分散的灯珠连成一个连通域，又不会像膨胀那样放大轮廓
    const cv::Mat mergeKernel =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * kMergeRadius + 1,
                                                              2 * kMergeRadius + 1));

    // ---------------- 第一遍：检测 + 状态机 ----------------
    std::vector<FrameResult> results;
    results.reserve(totalFrames);
    cv::Mat frame, hsv, warm, led, merged, vChannel, weighted;
    cv::Point2d smoothedCentre(0, 0);
    cv::Point2d lastCentre(0, 0);
    cv::Point2d centreVel(0, 0);
    bool haveCentre = false;

    int nextId = 1;
    int curId = -1;
    int lostCount = 0;
    int reselectCount = 0;
    int detectedFrames = 0;
    cv::Point2d predCentre(0, 0);   // 目标当前用于显示的圆心
    cv::Point2d targetVel(0, 0);    // 目标圆心的估计速度，用于匀速外推
    bool haveObs = false;           // predCentre 是否对应一次真实观测
    double predRadius = 0.0;
    int frameIdx = 0;

    while (frameIdx < totalFrames && cap.read(frame)) {
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        warmMask(hsv, warm);
        ledMask(hsv, led);
        cv::extractChannel(hsv, vChannel, 2);   // V 通道，用于亮度加权
        if (weighted.empty()) weighted = cv::Mat::zeros(frame.size(), CV_8U);

        FrameResult fr;
        fr.frame = frameIdx;

        // --- R 标中心 ---
        bool ok = false;
        const cv::Point2d c = weightedCentroid(weighted, vChannel, warm, 1500.0, ok);
        if (ok) {
            if (!haveCentre) {
                smoothedCentre = c;
                centreVel = {0, 0};
            } else {
                const cv::Point2d inst = c - lastCentre;
                centreVel = kVelocityEma * inst + (1 - kVelocityEma) * centreVel;
                smoothedCentre = kCenterEma * c + (1 - kCenterEma) * smoothedCentre;
            }
            lastCentre = c;
            haveCentre = true;
        } else if (haveCentre) {
            smoothedCentre += centreVel;   // 检测失败时按速度外推
        }
        fr.centreValid = ok;
        fr.centre = haveCentre ? smoothedCentre : cv::Point2d(0, 0);

        // --- 候选目标：合并同一扇叶的灯珠 ---
        cv::morphologyEx(led, merged, cv::MORPH_CLOSE, mergeKernel);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(merged, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        std::vector<Candidate> cands;
        for (size_t i = 0; i < contours.size(); ++i) {
            if (cv::contourArea(contours[i]) < kMinClusterArea) continue;
            // 只在包围盒内取该连通域包含的原始灯珠像素，再做圆拟合（避免整帧分配）
            const cv::Rect br = cv::boundingRect(contours[i]) &
                                cv::Rect(0, 0, frame.cols, frame.rows);
            if (br.width < 3 || br.height < 3) continue;
            cv::Mat small = cv::Mat::zeros(br.size(), CV_8U);
            std::vector<std::vector<cv::Point>> shifted(1);
            shifted[0].reserve(contours[i].size());
            for (const auto& q : contours[i]) shifted[0].push_back(q - br.tl());
            cv::drawContours(small, shifted, 0, cv::Scalar(255), cv::FILLED);
            cv::Mat pts;
            cv::bitwise_and(led(br), small, pts);
            std::vector<cv::Point> pix;
            cv::findNonZero(pts, pix);
            if (pix.size() < 20) continue;
            cv::Point2f cc;
            float rr = 0;
            cv::minEnclosingCircle(pix, cc, rr);
            Candidate cd;
            cd.centre = cv::Point2d(cc.x + br.x, cc.y + br.y);
            cd.radius = rr;
            cd.area = static_cast<double>(pix.size());
            cd.frame = frameIdx;
            cands.push_back(cd);
        }
        fr.candidates = cands;

        // --- 身份关联：候选点与「上一帧圆心的匀速外推」比较，只有落在门限内才继承原 ID ---
        const cv::Point2d predTarget = predCentre + targetVel;
        int matched = -1;
        double bestDist = 1e18;
        for (size_t i = 0; i < cands.size(); ++i) {
            const double d = cv::norm(cands[i].centre - predTarget);
            // 门限随半径缩放，避免扇叶较大时误拒
            const double gate = std::min(kAssocGate + 0.6 * cands[i].radius, kAssocGateMax);
            if (d < gate && d < bestDist) {
                bestDist = d;
                matched = static_cast<int>(i);
            }
        }

        if (curId < 0) {
            // 尚未锁定：首次出现有效目标时任选一个（取面积最大者）
            if (!cands.empty()) {
                size_t best = 0;
                for (size_t i = 1; i < cands.size(); ++i)
                    if (cands[i].area > cands[best].area) best = i;
                matched = static_cast<int>(best);
                curId = nextId++;
                lostCount = 0;
            }
        } else if (matched >= 0) {
            lostCount = 0;
        }

        if (curId >= 0 && matched >= 0) {
            fr.selected = matched;
            fr.targetId = curId;
            fr.detected = true;
            fr.lost = false;
            if (haveObs) {   // 只有拿到相邻两次真实观测才能估计速度
                targetVel = kVelocityEma * (cands[matched].centre - predCentre) +
                            (1.0 - kVelocityEma) * targetVel;
            } else {
                targetVel = {0, 0};
            }
            haveObs = true;
            predCentre = cands[matched].centre;
            // 半径做指数平滑：同一扇叶的灯珠时聚时散，平滑后轮廓不会来回跳
            predRadius = kRadiusEma * cands[matched].radius + (1.0 - kRadiusEma) * predRadius;
            if (predRadius <= 0.0) predRadius = cands[matched].radius;
            ++detectedFrames;
        } else if (curId >= 0) {
            // 丢失：保留身份，按原位置等待恢复
            ++lostCount;
            // 丢失期间保持最后一次观测位置（不做外推，避免预测跑飞）
            fr.targetId = curId;
            fr.detected = false;
            fr.lost = true;
            if (lostCount > kLostTolerance) {
                // 持续丢失超过容忍帧数 -> 放弃当前目标，允许重新选择
                curId = -1;
                lostCount = 0;
                ++reselectCount;
                predRadius = 0.0;
                targetVel = {0, 0};
                haveObs = false;
            }
        }
        fr.targetCircleCentre = predCentre;
        fr.targetRadius = predRadius;
        if (debug) {
            std::printf("f%3d ncand=%2zu | ", frameIdx, cands.size());
            for (const auto& cd : cands)
                std::printf("(%4.0f,%4.0f)r%3.0f a%5.0f  ", cd.centre.x, cd.centre.y, cd.radius,
                            cd.area);
            std::printf("|| matched=%d id=%d lost=%d pred=(%4.0f,%4.0f)",
                        matched, fr.targetId, lostCount, predCentre.x, predCentre.y);
            if (matched >= 0)
                std::printf(" dist=%.0f", cv::norm(cands[matched].centre - predTarget));
            std::printf("\n");
        }
        results.push_back(fr);
        ++frameIdx;
    }
    cap.release();

    if (results.empty()) {
        std::cerr << "视频没有帧\n";
        return 1;
    }

    // ---------------- 第二遍：渲染叠加视频 ----------------
    cv::VideoCapture cap2(inPath);
    cv::VideoWriter writer(outDir + "/recognition_overlay.mp4",
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                           cv::Size(width, height));
    cv::VideoWriter binWriter;
    if (writeBinary) {
        binWriter.open(outDir + "/binary_process.mp4",
                       cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                       cv::Size(width, height));
    }
    if (!writer.isOpened()) {
        std::cerr << "无法创建输出视频（检查 FFMPEG 后端）\n";
        return 1;
    }

    const cv::Scalar kWhite(255, 255, 255), kGreen(0, 255, 0), kRed(0, 0, 255);
    const cv::Scalar kYellow(0, 255, 255), kCyan(255, 255, 0), kGray(120, 120, 120);
    int idx = 0, written = 0;
    while (idx < static_cast<int>(results.size()) && cap2.read(frame)) {
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        warmMask(hsv, warm);
        ledMask(hsv, led);
        const FrameResult& fr = results[idx];
        cv::Mat vis = frame.clone();

        // 所有候选目标（细线，青色）
        for (const auto& cd : fr.candidates) {
            cv::circle(vis, cv::Point(cvRound(cd.centre.x), cvRound(cd.centre.y)),
                       cvRound(cd.radius), kCyan, 1, cv::LINE_AA);
        }

        // R 标中心
        const cv::Point centrePx(cvRound(fr.centre.x), cvRound(fr.centre.y));
        if (fr.centreValid) {
            cv::drawMarker(vis, centrePx, kWhite, cv::MARKER_CROSS, 34, 2, cv::LINE_AA);
            cv::circle(vis, centrePx, 12, kWhite, 1, cv::LINE_AA);
        }

        // 扇叶圆 + 圆心 + 连线
        const cv::Point tc(cvRound(fr.targetCircleCentre.x), cvRound(fr.targetCircleCentre.y));
        if (fr.targetId >= 0) {
            const bool okDraw = fr.detected || fr.lost;
            if (okDraw && fr.targetRadius > 0) {
                cv::circle(vis, tc, cvRound(fr.targetRadius), kGreen, 2, cv::LINE_AA);
                cv::circle(vis, tc + cv::Point(cvRound(-fr.targetRadius), 0), 3, kGreen,
                           cv::FILLED, cv::LINE_AA);
                cv::circle(vis, tc + cv::Point(cvRound(fr.targetRadius), 0), 3, kGreen,
                           cv::FILLED, cv::LINE_AA);
            }
            if (fr.detected) {
                cv::circle(vis, tc, 6, kGreen, cv::FILLED, cv::LINE_AA);
            } else {
                cv::drawMarker(vis, tc, kRed, cv::MARKER_TILTED_CROSS, 26, 2, cv::LINE_AA);
            }
            if (fr.centreValid) {
                cv::line(vis, centrePx, tc, kYellow, 2, cv::LINE_AA);
            }
            // 目标 ID
            cv::putText(vis, "ID " + std::to_string(fr.targetId),
                        tc + cv::Point(16, -16), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        fr.detected ? kGreen : kRed, 2, cv::LINE_AA);
        }

        // HUD
        double angleDeg = 0.0;
        if (fr.centreValid && fr.targetId >= 0) {
            const double th = std::atan2(fr.centre.y - fr.targetCircleCentre.y,
                                         fr.targetCircleCentre.x - fr.centre.x);
            angleDeg = th * 180.0 / kPi;
        }
        std::vector<std::string> lines;
        std::vector<cv::Scalar> colors;
        lines.push_back("task 3 - power rune recognition & stable tracking");
        colors.push_back(kWhite);
        lines.push_back("frame " + std::to_string(fr.frame) + " / " +
                        std::to_string(static_cast<int>(results.size())) + "   t = " +
                        fmt("%.3f", fr.frame / fps) + " s");
        colors.push_back(kWhite);
        lines.push_back("R mark centre (" + fmt("%.0f", fr.centre.x) + ", " +
                        fmt("%.0f", fr.centre.y) + ") px" +
                        (fr.centreValid ? "" : "  (predicted)"));
        colors.push_back(fr.centreValid ? kWhite : kGray);
        lines.push_back("blade circle centre (" + fmt("%.0f", fr.targetCircleCentre.x) + ", " +
                        fmt("%.0f", fr.targetCircleCentre.y) + ") px   r = " +
                        fmt("%.0f", fr.targetRadius) + " px");
        colors.push_back(kGreen);
        lines.push_back("target angle = " + fmt("%.2f", angleDeg) + " deg    candidates = " +
                        std::to_string(fr.candidates.size()));
        colors.push_back(kYellow);
        if (fr.targetId < 0) {
            lines.push_back("tracking status: SEARCHING");
            colors.push_back(kCyan);
        } else if (fr.detected) {
            lines.push_back("tracking status: DETECTED  (ID " + std::to_string(fr.targetId) +
                            ")");
            colors.push_back(kGreen);
        } else {
            lines.push_back("tracking status: LOST  (ID " + std::to_string(fr.targetId) +
                            ", waiting for recovery)");
            colors.push_back(kRed);
        }
        drawHud(vis, lines, colors, cv::Point(12, 12));
        writer.write(vis);
        ++written;

        // 中间二值化过程
        if (binWriter.isOpened()) {
            cv::Mat bin(frame.size(), CV_8UC3, cv::Scalar(0, 0, 0));
            bin.setTo(cv::Scalar(40, 40, 90), warm);   // 机关本体：暗红
            bin.setTo(cv::Scalar(0, 170, 255), led);   // 灯珠：橙色
            for (const auto& cd : fr.candidates) {
                cv::circle(bin, cv::Point(cvRound(cd.centre.x), cvRound(cd.centre.y)),
                           cvRound(cd.radius), kCyan, 1, cv::LINE_AA);
            }
            if (fr.centreValid) cv::drawMarker(bin, centrePx, kWhite, cv::MARKER_CROSS, 34, 2,
                                               cv::LINE_AA);
            cv::putText(bin, "binary: warm mask (dark red) + LED mask (orange)",
                        cv::Point(16, height - 24), cv::FONT_HERSHEY_SIMPLEX, 0.6, kWhite, 1,
                        cv::LINE_AA);
            binWriter.write(bin);
        }
        ++idx;
    }
    writer.release();
    if (binWriter.isOpened()) binWriter.release();
    cap2.release();

    // ---------------- 文本报告 ----------------
    std::ostringstream rep;
    rep << "视频: " << inPath << "\n"
        << "分辨率/帧率: " << width << "x" << height << " / " << fps << " FPS\n"
        << "总帧数: " << results.size() << "\n"
        << "有效观测帧: " << detectedFrames << " (" << fmt("%.1f", 100.0 * detectedFrames / results.size())
        << "%)\n"
        << "丢失帧: " << (static_cast<int>(results.size()) - detectedFrames) << "\n"
        << "重选次数: " << reselectCount << "\n"
        << "丢失容忍: " << kLostTolerance << " 帧 (" << fmt("%.2f", kLostTolerance / fps) << " s)\n"
        << "关联门限: " << kAssocGate << " px + 0.6*r\n";
    std::cout << rep.str();
    std::ofstream(outDir + "/tracking_report.txt") << rep.str();
    std::cout << "输出: " << outDir << "/recognition_overlay.mp4 (" << written << " 帧)"
              << (writeBinary ? " + binary_process.mp4" : "") << "\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    // 默认按讲义要求处理两个视频；也可用命令行指定单个视频
    if (argc >= 3) {
        const int maxFrames = argc >= 4 ? std::atoi(argv[3]) : 0;
        return processVideo(argv[1], argv[2], true, maxFrames);
    }
    int rc = 0;
    rc |= processVideo("resources/task_3.mp4", "result/task3_windmill/task_3", true);
    rc |= processVideo("resources/task_4.mp4", "result/task3_windmill/task_4", true);
    return rc;
}
