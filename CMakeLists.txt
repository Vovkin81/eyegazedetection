cmake_minimum_required(VERSION 3.0.0)
project(eyegazedetection VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(eyegazedetection main.cpp)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_PREFIX_PATH "~/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

target_link_libraries(eyegazedetection PRIVATE Threads::Threads dlib::dlib ${OpenCV_LIBS} ${TORCH_LIBRARIES})

find_package(dlib REQUIRED)

set_property(TARGET eyegazedetection PROPERTY CXX_STANDARD 14)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
