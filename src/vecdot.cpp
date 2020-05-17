/**
* @author : Milinda Fernando (milinda@cs.utah.edu)
* @brief : Perform vector dot product using OneAPI MKL library. 
*
*/

#include <vector>
#include <iostream>
#include <CL/sycl.hpp>
#include "mkl.h"
#include "mkl_blas_sycl.hpp"

#define SIZE 1024
namespace sycl = cl::sycl;

#define VECType double

int main() 
{
    std::array<VECType, SIZE> vec_a;
    std::array<VECType, SIZE> vec_b;

    VECType dot_ab_dev  = 0.0;
    VECType dot_ab_host = 0.0;

    for (int i = 0; i<SIZE; ++i)
    {
        vec_a[i] = 1.0;
        vec_b[i] = 1.0;
    }


    std::cout<<"================================================================================="<<std::endl;
    std::cout<<"             Computing vector dot product on the host(CPU)                       "<<std::endl;
    std::cout<<"================================================================================="<<std::endl;

    for (int i = 0; i<SIZE; ++i)
        dot_ab_host = dot_ab_host + (vec_a[i]*vec_b[i]);

    
    std::cout<<"dot(vec_a,vec_b): "<<dot_ab_host<<std::endl;

    std::cout<<"================================================================================="<<std::endl;
    std::cout<<"             Computing vector dot product on the device(GPU)                     "<<std::endl;
    std::cout<<"================================================================================="<<std::endl;


    auto asyncHandler = [&](cl::sycl::exception_list eL) {
    for (auto &e : eL) {
      try {
        std::rethrow_exception(e);
      } catch (cl::sycl::exception &e) {
        std::cout << e.what() << std::endl;
        std::cout << "fail" << std::endl;
        // std::terminate() will exit the process, return non-zero, and output a
        // message to the user about the exception
        std::terminate();
      }
    }
    };

    try
    {

        sycl::default_selector gpu;
        sycl::queue d_queue(gpu,asyncHandler);

        auto a_size = sycl::range<1>{vec_a.size()};
        auto b_size = sycl::range<1>{vec_b.size()};
        auto dot_ab_size = sycl::range<1>{1};

        assert(a_size==b_size);

        sycl::buffer<VECType, 1>  _vec_a(vec_a.data(), a_size);
        sycl::buffer<VECType, 1>  _vec_b(vec_b.data(), b_size);
        sycl::buffer<VECType, 1>  _dot_ab_dev(&dot_ab_dev, dot_ab_size);

        mkl::blas::dot(d_queue, vec_a.size(), _vec_a, 1, _vec_b, 1, _dot_ab_dev);



    }catch (cl::sycl::exception const &e) 
    {
        std::cout << "\t\tSYCL exception during GEMM\n"
              << e.what() << std::endl
              << "OpenCL status: " << e.get_cl_code() << std::endl;
    }
    

    std::cout<<"dot(vec_a,vec_b): "<<dot_ab_dev<<std::endl;





}