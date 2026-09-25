cmake_minimum_required(VERSION 3.16)
project(vision_training LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)

# ------------------------ 任务 1：OpenCV 图像处理 ------------------------
add_executable(task1_image src/task1_image/main.cpp)
target_include_directories(task1_image PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(task1_image PRIVATE ${OpenCV_LIBS})

# --------------------- 任务 2：合成旋转视频参数拟合 ---------------------
# Eigen 用于线性最小二乘（列主元 QR），Ceres 用于非线性精修
add_executable(task2_fit src/task2_fit/main.cpp)
target_include_directories(task2_fit PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(task2_fit PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
    Ceres::ceres)
