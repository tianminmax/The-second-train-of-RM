#pragma once

// ---------------------------------------------------------------------------
//  轻量绘图工具（header-only）
//
//  把 (x, y) 数据画成曲线 / 散点图并保存为 PNG，供任务 2、任务 3 共用，
//  避免每个任务各写一份绘图代码。只依赖 OpenCV 的画布与 Hershey 字体。
//
//  典型用法：
//      plot::Figure fig(1400, 800, "title", "t (s)", "y");
//      fig.setRange(0, 24, -1, 1);
//      fig.drawAxes();
//      fig.drawPoints(observed, cv::Scalar(200, 120, 0), 3, 8);
//      fig.drawCurve(fitted, cv::Scalar(0, 0, 220), 2);
//      fig.drawLegend({{"observation", cv::Scalar(200, 120, 0)},
//                      {"fit", cv::Scalar(0, 0, 220)}});
//      fig.save("result/xxx.png");
// ---------------------------------------------------------------------------

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace plot {

// 取一个"整齐"的刻度间隔：1 / 2 / 5 × 10^k
inline double niceStep(double range, int targetDivisions) {
    if (range <= 0.0 || targetDivisions < 1) return 1.0;
    const double raw = range / targetDivisions;
    const double mag = std::pow(10.0, std::floor(std::log10(raw)));
    const double norm = raw / mag;
    const double step =
        (norm <= 1.0 ? 1.0 : norm <= 2.0 ? 2.0 : norm <= 5.0 ? 5.0 : 10.0) * mag;
    return step;
}

// 按刻度间隔决定小数位数，避免出现 "-0.0"
inline std::string tickLabel(double v, double step) {
    if (std::fabs(v) < step * 1e-6) v = 0.0;
    int decimals = 0;
    for (double s = step; s < 1.0 && decimals < 6; s *= 10.0) ++decimals;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
    return std::string(buf);
}

// 竖排文字，用于 y 轴标签
inline void putVerticalText(cv::Mat& img, const std::string& text, cv::Point org,
                            double scale, const cv::Scalar& color, int thickness) {
    int baseline = 0;
    const cv::Size sz =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
    cv::Mat tmp(sz.height + baseline + 6, sz.width + 6, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::putText(tmp, text, cv::Point(3, sz.height + 3), cv::FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv::LINE_AA);
    cv::rotate(tmp, tmp, cv::ROTATE_90_COUNTERCLOCKWISE);
    if (org.x >= 0 && org.y >= 0 && org.x + tmp.cols <= img.cols &&
        org.y + tmp.rows <= img.rows) {
        tmp.copyTo(img(cv::Rect(org.x, org.y, tmp.cols, tmp.rows)));
    }
}

class Figure {
public:
    Figure(int width, int height, const std::string& title, const std::string& xlabel,
           const std::string& ylabel)
        : canvas_(height, width, CV_8UC3, cv::Scalar(255, 255, 255)),
          w_(width),
          h_(height),
          title_(title),
          xlabel_(xlabel),
          ylabel_(ylabel) {
        cv::putText(canvas_, title_, cv::Point(padLeft_, 34), cv::FONT_HERSHEY_SIMPLEX,
                    0.62, cv::Scalar(20, 20, 20), 2, cv::LINE_AA);
    }

    void setPadding(int left, int right, int top, int bottom) {
        padLeft_ = left;
        padRight_ = right;
        padTop_ = top;
        padBottom_ = bottom;
    }

    void setRange(double xlo, double xhi, double ylo, double yhi) {
        xmin_ = xlo;
        xmax_ = xhi;
        ymin_ = ylo;
        ymax_ = yhi;
    }

    cv::Point toPixel(double x, double y) const {
        const double px = (x - xmin_) / (xmax_ - xmin_);
        const double py = (ymax_ - y) / (ymax_ - ymin_);
        return cv::Point(cvRound(padLeft_ + px * plotWidth()),
                         cvRound(padTop_ + py * plotHeight()));
    }

    // 画网格、刻度、边框和轴标签
    void drawAxes() {
        const cv::Rect area = plotArea();
        const double xs = niceStep(xmax_ - xmin_, 8);
        const double ys = niceStep(ymax_ - ymin_, 6);

        for (double x = std::ceil(xmin_ / xs) * xs; x <= xmax_ + xs * 1e-6; x += xs) {
            const int px = toPixel(x, ymin_).x;
            cv::line(canvas_, cv::Point(px, area.y), cv::Point(px, area.y + area.height),
                     cv::Scalar(232, 232, 232), 1, cv::LINE_AA);
            cv::putText(canvas_, tickLabel(x, xs), cv::Point(px - 22, area.y + area.height + 22),
                        cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(60, 60, 60), 1, cv::LINE_AA);
        }
        for (double y = std::ceil(ymin_ / ys) * ys; y <= ymax_ + ys * 1e-6; y += ys) {
            const int py = toPixel(xmin_, y).y;
            cv::line(canvas_, cv::Point(area.x, py), cv::Point(area.x + area.width, py),
                     cv::Scalar(232, 232, 232), 1, cv::LINE_AA);
            const std::string s = tickLabel(y, ys);
            int baseline = 0;
            const cv::Size sz =
                cv::getTextSize(s, cv::FONT_HERSHEY_SIMPLEX, 0.42, 1, &baseline);
            cv::putText(canvas_, s, cv::Point(area.x - sz.width - 10, py + 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(60, 60, 60), 1, cv::LINE_AA);
        }
        cv::rectangle(canvas_, area, cv::Scalar(130, 130, 130), 1, cv::LINE_AA);
        cv::putText(canvas_, xlabel_,
                    cv::Point(area.x + area.width / 2 - 30, h_ - 18),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(20, 20, 20), 1, cv::LINE_AA);
        putVerticalText(canvas_, ylabel_,
                        cv::Point(14, area.y + std::max(0, area.height / 2 - 70)), 0.5,
                        cv::Scalar(20, 20, 20), 1);
    }

    // 散点（观测），stride 用于抽稀
    void drawPoints(const std::vector<cv::Point2d>& pts, const cv::Scalar& color, int radius,
                    int stride) {
        const int step = std::max(1, stride);
        for (size_t i = 0; i < pts.size(); i += step) {
            const cv::Point p = toPixel(pts[i].x, pts[i].y);
            if (p.x < 0 || p.y < 0 || p.x >= w_ || p.y >= h_) continue;
            cv::circle(canvas_, p, radius, color, cv::FILLED, cv::LINE_AA);
        }
    }

    // 折线（拟合曲线）
    void drawCurve(const std::vector<cv::Point2d>& pts, const cv::Scalar& color,
                   int thickness) {
        std::vector<cv::Point> poly;
        poly.reserve(pts.size());
        for (const auto& q : pts) poly.push_back(toPixel(q.x, q.y));
        if (poly.size() > 1) cv::polylines(canvas_, poly, false, color, thickness, cv::LINE_AA);
    }

    void drawHorizontalLine(double y, const cv::Scalar& color, int thickness) {
        const cv::Rect area = plotArea();
        const int py = toPixel(xmin_, y).y;
        cv::line(canvas_, cv::Point(area.x, py), cv::Point(area.x + area.width, py), color,
                 thickness, cv::LINE_AA);
    }

    void drawLegend(const std::vector<std::pair<std::string, cv::Scalar>>& items) {
        if (items.empty()) return;
        int baseline = 0;
        int maxw = 0;
        for (const auto& it : items) {
            maxw = std::max(maxw, cv::getTextSize(it.first, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1,
                                                  &baseline)
                                     .width);
        }
        const int lineH = 26;
        const int boxW = maxw + 80;
        const int boxH = lineH * static_cast<int>(items.size()) + 16;
        const cv::Rect box(w_ - padRight_ - boxW - 14, padTop_ + 14, boxW, boxH);
        cv::rectangle(canvas_, box, cv::Scalar(255, 255, 255), cv::FILLED);
        cv::rectangle(canvas_, box, cv::Scalar(150, 150, 150), 1, cv::LINE_AA);
        for (size_t i = 0; i < items.size(); ++i) {
            const int y = box.y + 18 + static_cast<int>(i) * lineH;
            cv::line(canvas_, cv::Point(box.x + 14, y - 4), cv::Point(box.x + 52, y - 4),
                     items[i].second, 3, cv::LINE_AA);
            cv::putText(canvas_, items[i].first, cv::Point(box.x + 60, y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(30, 30, 30), 1,
                        cv::LINE_AA);
        }
    }

    cv::Mat& canvas() { return canvas_; }
    const cv::Mat& canvas() const { return canvas_; }
    bool save(const std::string& path) const { return cv::imwrite(path, canvas_); }

private:
    int plotWidth() const { return w_ - padLeft_ - padRight_; }
    int plotHeight() const { return h_ - padTop_ - padBottom_; }
    cv::Rect plotArea() const {
        return cv::Rect(padLeft_, padTop_, plotWidth(), plotHeight());
    }

    cv::Mat canvas_;
    int w_, h_;
    int padLeft_ = 95, padRight_ = 35, padTop_ = 60, padBottom_ = 70;
    double xmin_ = 0.0, xmax_ = 1.0, ymin_ = 0.0, ymax_ = 1.0;
    std::string title_, xlabel_, ylabel_;
};

}  // namespace plot
