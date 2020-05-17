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

#define SIZE 32
namespace sycl = cl::sycl;

#define VECType double

int main() 
{
    std::array<VECType, SIZE*SIZE> A;
    std::array<VECType, SIZE*SIZE> B;
    std::array<VECType, SIZE*SIZE> C_dev;
    std::array<VECType, SIZE*SIZE> C_host;


    // C= A*B
    for (unsigned int i = 0; i<SIZE; ++i)
    {
        for (unsigned int j = 0; j<SIZE; ++j)
        {
            A[i*SIZE + j] = i + j;
            B[i*SIZE + j] = i + j;
            C_dev[i*SIZE + j] = 0.0;
            C_host[i*SIZE + j] = 0.0;
        }
          
    }

    std::cout<<"================================================================================="<<std::endl;
    std::cout<<"             Computing mat-mat product on the host(CPU)                       "<<std::endl;
    std::cout<<"================================================================================="<<std::endl;

    for (unsigned int i = 0; i<SIZE; ++i)
      for(unsigned int j = 0; j < SIZE; ++j)
       for(unsigned int k=0; k < SIZE; k++)
        C_host[i*SIZE + j] += A[i*SIZE + k]  + B[k*SIZE + j];

    
    std::cout<<"================================================================================="<<std::endl;
    std::cout<<"             Computing mat-mat product on the device(GPU)                     "<<std::endl;
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

        auto A_size = sycl::range<1>{A.size()};
        auto B_size = sycl::range<1>{B.size()};
        auto C_size = sycl::range<1>{C_dev.size()};

        sycl::buffer<VECType, 1>  _A(A.data(), A_size);
        sycl::buffer<VECType, 1>  _B(B.data(), B_size);
        sycl::buffer<VECType, 1>  _C(C_dev.data(), C_size);

        mkl::blas::gemm(d_queue, mkl::transpose::nontrans, mkl::transpose::nontrans,  SIZE, SIZE, SIZE, 1.0, _A, SIZE, _B, SIZE, 0.0 , _C, SIZE);



    }catch (cl::sycl::exception const &e) 
    {
        std::cout << "\t\tSYCL exception during GEMM\n"
              << e.what() << std::endl
              << "OpenCL status: " << e.get_cl_code() << std::endl;
    }
    

    //std::cout<<"dot(vec_a,vec_b): "<<dot_ab_dev<<std::endl;





}