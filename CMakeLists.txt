cmake_minimum_required(VERSION 2.8)
find_package(OpenCV REQUIRED)
project(lane_detection)

set (source_files
	main.cpp
	api_lane_detection.h
	api_lane_detection.cpp
	msac/MSAC.h
	msac/MSAC.cpp
	msac/errorNIETO.h
	msac/errorNIETO.cpp
	msac/lmmin.h
	msac/lmmin.cpp

)

set( lib_name vanishing-point)
add_library( ${lib_name} ${source_files} )
target_link_libraries( ${lib_name} ${OpenCV_LIBS})

set( test_exec_name maintest)
add_executable(${test_exec_name} main.cpp)

target_link_libraries( ${test_exec_name} ${lib_name} ${OpenCV_LIBS})
target_link_libraries( ${test_exec_name} ${OpenCV_LIBS})
