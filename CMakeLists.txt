cmake_minimum_required(VERSION 3.16)
project(task1_image LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(task1_image src/task1_image/main.cpp)
target_include_directories(task1_image PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(task1_image PRIVATE ${OpenCV_LIBS})