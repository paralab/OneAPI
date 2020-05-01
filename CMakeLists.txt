CMAKE_MINIMUM_REQUIRED_VERSION(VERSION 2.8.9)
PROJECT(OneAPITest)

# set both C and C++ compiler to dpcpp explicitly. 
SET(CC dpcpp)
SET(CXX dpcpp)

SET(INC_FILES include/benchOneApi.h)

add_executable(benchOneAPI src/benchOneAPI.cpp)
target_include_directories(benchOneAPI PRIVATE ${CMAKE_SOURCE_DIR}/include)