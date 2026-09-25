#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    // ============================================================
    // 第1步：读图与颜色转换
    // ============================================================
    cv::Mat img = cv::imread("resources/test_image.jpg");
    if (img.empty()) {
        std::cerr << "读图失败，检查 test_image.jpg 路径！" << std::endl;
        return 1;
    }
    std::cout << "原图: " << img.cols << "x" << img.rows
              << ", channels=" << img.channels() << std::endl;

    // 灰度图
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::imwrite("gray.png", gray);

    // HSV 图
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    // 拆分 H、S、V
    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv, hsv_channels);
    cv::imwrite("hsv_h.png", hsv_channels[0]);
    cv::imwrite("hsv_s.png", hsv_channels[1]);
    cv::imwrite("hsv_v.png", hsv_channels[2]);

    std::cout << "第1步完成: gray.png / hsv_h.png / hsv_s.png / hsv_v.png" << std::endl;

    // ============================================================
    // 第2步：滤波对比
    // ============================================================
    cv::Mat meanImg, gaussianImg, medianImg;

    // 均值滤波 5x5
    cv::blur(img, meanImg, cv::Size(5, 5));
    cv::imwrite("mean_filter.png", meanImg);

    // 高斯滤波 5x5, sigmaX=1.5
    cv::GaussianBlur(img, gaussianImg, cv::Size(5, 5), 1.5);
    cv::imwrite("gaussian_filter.png", gaussianImg);

    // 中值滤波 核=5
    cv::medianBlur(img, medianImg, 5);
    cv::imwrite("median_filter.png", medianImg);

    std::cout << "第2步完成: mean_filter.png / gaussian_filter.png / median_filter.png" << std::endl;

    // ============================================================
    // 第3步：红色提取（HSV 双区间）
    // ============================================================
    // 红色 H 在 0 附近和 180 附近
    cv::Mat mask_low, mask_high, red_mask;
    cv::inRange(hsv, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), mask_low);
    cv::inRange(hsv, cv::Scalar(170, 100, 100), cv::Scalar(179, 255, 255), mask_high);
    red_mask = mask_low | mask_high;
    cv::imwrite("red_mask.png", red_mask);

    std::cout << "第3步完成: red_mask.png" << std::endl;

    // ============================================================
    // 第4步：形态学 + 轮廓提取 + 筛选
    // ============================================================
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    cv::Mat eroded, dilated, opened, closed;
    cv::erode(red_mask, eroded, kernel);
    cv::dilate(red_mask, dilated, kernel);
    cv::morphologyEx(red_mask, opened, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(red_mask, closed, cv::MORPH_CLOSE, kernel);

    cv::imwrite("erode.png", eroded);
    cv::imwrite("dilate.png", dilated);
    cv::imwrite("open.png", opened);
    cv::imwrite("close.png", closed);

    // 用开运算后的结果找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(opened, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat contour_result = img.clone();
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area < 500.0) continue;

        cv::Rect box = cv::boundingRect(contours[i]);
        double ratio = static_cast<double>(box.width) / box.height;
        if (ratio < 0.2 || ratio > 5.0) continue;

        cv::rectangle(contour_result, box, cv::Scalar(0, 0, 255), 2);
        cv::drawContours(contour_result, contours, static_cast<int>(i),
                         cv::Scalar(0, 255, 0), 2);

        std::cout << "轮廓 " << i << " 面积=" << area
                  << " 外接矩形=" << box << std::endl;
    }
    cv::imwrite("contours_boxes.png", contour_result);

    std::cout << "第4步完成: erode/dilate/open/close/contours_boxes.png" << std::endl;

    // ============================================================
    // 第5步：绘制与变换
    // ============================================================
    // 5.1 在原图副本上画圆、矩形、文字
    cv::Mat draw_img = img.clone();
    cv::circle(draw_img, cv::Point(200, 200), 80, cv::Scalar(0, 255, 0), 3);
    cv::rectangle(draw_img, cv::Rect(300, 100, 150, 100), cv::Scalar(255, 0, 0), 3);
    cv::putText(draw_img, "OpenCV Demo", cv::Point(50, 50),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    cv::imwrite("drawing.png", draw_img);

    // 5.2 绕图像中心旋转 35 度
    cv::Point2f center(img.cols / 2.0f, img.rows / 2.0f);
    cv::Mat rot = cv::getRotationMatrix2D(center, 35.0, 1.0);
    cv::Mat rotated;
    cv::warpAffine(img, rotated, rot, img.size());
    cv::imwrite("rotated_35deg.png", rotated);

    // 5.3 裁剪左上角 1/4
    int w = img.cols / 2;
    int h = img.rows / 2;
    cv::Mat crop = img(cv::Rect(0, 0, w, h));
    cv::imwrite("crop_top_left.png", crop);

    std::cout << "第5步完成: drawing.png / rotated_35deg.png / crop_top_left.png" << std::endl;

    return 0;
}