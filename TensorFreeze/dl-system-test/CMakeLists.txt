cmake_minimum_required(VERSION 3.10)
project(boost_cc)
set(CMAKE_CXX_STANDARD 11)
include_directories(/opt/intel/mkl/include)
add_executable(boost_cc boost_cc.cpp)

