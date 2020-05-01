/**
 * @file benchOneAPI.h
 * @brief simple benchmark to test dot mat-vec and mat-mat multiplications. 
 * @version 0.1
 * @date 2020-05-01
 * @copyright Copyright (c) 2020
 * 
 */

#pragma once
#include <vector>
#include <CL/sycl.hpp>

#define SIZE 1024
namespace sycl = cl::sycl;