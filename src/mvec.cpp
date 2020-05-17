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
    std::array<VECType, SIZE*SIZE> M;
    std::array<VECType, SIZE> u;

    std::array<VECType, SIZE> v_host;
    std::array<VECType, SIZE> v_dev;
    
    for (unsigned int i = 0; i<SIZE; ++i)
    {
        u[i]=1.0;
        v_host[i] = 0;
        v_dev[i]  = 0;
        
        for (unsigned int j = 0; j<SIZE; ++j)
          M[i*SIZE + j] = i + j;
    }

    std::cout<<"================================================================================="<<std::endl;
    std::cout<<"             Computing matvec on the host(CPU)                       "<<std::endl;
    std::cout<<"================================================================================="<<std::endl;

    for (unsigned int i = 0; i<SIZE; ++i)
      for(unsigned int j = 0; j < SIZE; ++j)
        v_host[i] += M[i*SIZE + j ] * u[j];
    
    //std::cout<<"dot(vec_a,vec_b): "<<dot_ab_host<<std::endl;

    std::cout<<"================================================================================="<<std::endl;
    std::cout<<"             Computing matvec on the device(GPU)                     "<<std::endl;
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

        auto M_size = sycl::range<1>{M.size()};
        auto u_size = sycl::range<1>{u.size()};
        auto v_size = sycl::range<1>{v_dev.size()};

        assert(u_size == v_size);

        sycl::buffer<VECType, 1>  _M(M.data(), M_size);
        sycl::buffer<VECType, 1>  _u(u.data(), u_size);
        sycl::buffer<VECType, 1>  _v(v_dev.data(), v_size);

        mkl::blas::gemv(d_queue, mkl::transpose::nontrans, SIZE, SIZE, 1.0, _M, SIZE, _u, 1, 0.0 , _v, 1);



    }catch (cl::sycl::exception const &e) 
    {
        std::cout << "\t\tSYCL exception during GEMM\n"
              << e.what() << std::endl
              << "OpenCL status: " << e.get_cl_code() << std::endl;
    }
    

    //std::cout<<"dot(vec_a,vec_b): "<<dot_ab_dev<<std::endl;





}