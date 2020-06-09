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
#include <chrono>
#include <ctime>

namespace sycl = cl::sycl;

#define VECType double

int main(int argc, char** argv) 
{

    if(argc < 2)
    {
      std::cout<<"Usage: "<<argv[0]<<" mat(sz) iter"<<std::endl;
      exit(0);
    }  

    const unsigned int SIZE = atoi(argv[1]);
    const unsigned int ITER   = atoi(argv[2]); 
    double tick_count;

    std::vector<VECType> A;
    std::vector<VECType> B;
    std::vector<VECType> C_dev;
    std::vector<VECType> C_host;

    A.resize(SIZE*SIZE);
    B.resize(SIZE*SIZE);
    C_dev.resize(SIZE*SIZE);
    C_host.resize(SIZE*SIZE);


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

    auto t_start = std::chrono::high_resolution_clock::now();

    for(unsigned int it=0; it < ITER; it++)
    {

      for (unsigned int i = 0; i<SIZE; ++i)
      for(unsigned int j = 0; j < SIZE; ++j)
      {
        C_host[i*SIZE + j] = 0;

        for(unsigned int k=0; k < SIZE; k++)
          C_host[i*SIZE + j] += A[i*SIZE + k]  + B[k*SIZE + j];
      }
       


    }

    
    auto t_end = std::chrono::high_resolution_clock::now();
    tick_count=(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
    std::cout<<"time host (s): "<<(tick_count/((double)1000))<<std::endl;
    
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

    t_start = std::chrono::high_resolution_clock::now();
    
    using wall_clock_t = std::chrono::high_resolution_clock;
    using time_point_t = std::chrono::high_resolution_clock::time_point;
    std::vector<time_point_t> eventList(2);
    std::vector<time_point_t> startTimeList(2);


    try
    {

        sycl::gpu_selector gpu;
        sycl::queue d_queue(gpu,asyncHandler);
        std::cout << "Device: "<< d_queue.get_device().get_info<sycl::info::device::name>()<< std::endl;
        std::cout << "global mem: "<<d_queue.get_device().get_info<sycl::info::device::global_mem_size>()<<std::endl;
        std::cout << "local  mem: "<<d_queue.get_device().get_info<sycl::info::device::local_mem_size>()<<std::endl;
        //std::cout << "local  mem: "<<d_queue.get_device().get_info<sycl::info::device::double_fp_config>()<<std::endl;

        auto A_size = sycl::range<1>{A.size()};
        auto B_size = sycl::range<1>{B.size()};
        auto C_size = sycl::range<1>{C_dev.size()};

        sycl::buffer<VECType, 1>  _A(A.data(), A_size);
        sycl::buffer<VECType, 1>  _B(B.data(), B_size);
        sycl::buffer<VECType, 1>  _C(C_dev.data(), C_size);

        startTimeList.at(0) = wall_clock_t::now();

        for(unsigned int it=0; it< ITER; it++)
          mkl::blas::gemm(d_queue, mkl::transpose::nontrans, mkl::transpose::nontrans,  SIZE, SIZE, SIZE, 1.0, _A, SIZE, _B, SIZE, 0.0 , _C, SIZE);

        d_queue.wait();
        startTimeList.at(1) = wall_clock_t::now();



    }catch (cl::sycl::exception const &e) 
    {
        std::cout << "\t\tSYCL exception during GEMM\n"
              << e.what() << std::endl
              << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    // std::cout<<"time device kernel execution (mu s): "<<std::chrono::duration_cast<std::chrono::microseconds>(startTimeList[1]-startTimeList[0]).count()<<std::endl;

    t_end = std::chrono::high_resolution_clock::now();
    tick_count=(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
    std::cout<<"time device (s): "<<(tick_count/((double)1000))<<std::endl;

    //std::cout<<"dot(vec_a,vec_b): "<<dot_ab_dev<<std::endl;





}