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

    const unsigned int MAT_SZ = atoi(argv[1]);
    const unsigned int ITER   = atoi(argv[2]); 
    double tick_count;

    const unsigned int SIZE = MAT_SZ;
    std::vector<VECType> M;
    std::vector<VECType> u;

    std::vector<VECType> v_host;
    std::vector<VECType> v_dev;

    M.resize(MAT_SZ*MAT_SZ);
    u.resize(SIZE);

    v_dev.resize(SIZE);
    v_host.resize(SIZE);
    
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
    
    auto t_start = std::chrono::high_resolution_clock::now();

    for(unsigned int it=0; it< ITER; it++)
    {

      for (unsigned int i = 0; i<SIZE; ++i)
      {
          v_host[i]=0;
          for(unsigned int j = 0; j < SIZE; ++j)
            v_host[i] += M[i*SIZE + j ] * u[j];
      }
        
    }

    
    auto t_end = std::chrono::high_resolution_clock::now();
    //std::cout<<"time host (mu s): "<<std::chrono::duration_cast<std::chrono::microseconds>(t_end-t_start).count()<<std::endl;
    //std::cout<<"dot(vec_a,vec_b): "<<dot_ab_host<<std::endl;
    tick_count=(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
    std::cout<<"time host (s): "<<(tick_count/((double)1000))<<std::endl;

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

    t_start = std::chrono::high_resolution_clock::now();
    
    using wall_clock_t = std::chrono::high_resolution_clock;
    using time_point_t = std::chrono::high_resolution_clock::time_point;
    std::vector<time_point_t> eventList(2);
    std::vector<time_point_t> startTimeList(2);
    
    try
    {

        // sycl::gpu_selector gpu;
        // sycl::property_list propList{sycl::property::queue::enable_profiling()};
        // sycl::queue d_queue(gpu,asyncHandler,propList);
        sycl::gpu_selector gpu;
        sycl::queue d_queue(gpu,asyncHandler);
        std::cout << "Device: "<< d_queue.get_device().get_info<sycl::info::device::name>()<< std::endl;
        std::cout << "global mem: "<<d_queue.get_device().get_info<sycl::info::device::global_mem_size>()<<std::endl;
        std::cout << "local  mem: "<<d_queue.get_device().get_info<sycl::info::device::local_mem_size>()<<std::endl;



        auto M_size = sycl::range<1>{M.size()};
        auto u_size = sycl::range<1>{u.size()};
        auto v_size = sycl::range<1>{v_dev.size()};

        assert(u_size == v_size);
    

        sycl::buffer<VECType, 1>  _M(M.data(), M_size);
        sycl::buffer<VECType, 1>  _u(u.data(), u_size);
        sycl::buffer<VECType, 1>  _v(v_dev.data(), v_size);

        startTimeList.at(0) = wall_clock_t::now();
        
        for(unsigned int it=0; it< ITER; it++)
          mkl::blas::gemv(d_queue, mkl::transpose::nontrans, SIZE, SIZE, 1.0, _M, SIZE, _u, 1, 0.0 , _v, 1);

        d_queue.wait();
        startTimeList.at(1) = wall_clock_t::now();



    }catch (cl::sycl::exception const &e) 
    {
        std::cout << "\t\tSYCL exception during GEMM\n"
              << e.what() << std::endl
              << "OpenCL status: " << e.get_cl_code() << std::endl;
    }
    
    //sycl::sycl_profiler profiler(eventList, startTimeList);
    //std::cout << "Kernel exection:t" << profiler.get_kernel_execution_time() << std::endl;
    // std::cout<<"time device kernel execution (mu s): "<<std::chrono::duration_cast<std::chrono::microseconds>(startTimeList[1]-startTimeList[0]).count()<<std::endl;

    t_end = std::chrono::high_resolution_clock::now();
    tick_count=(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
    std::cout<<"time device (s): "<<(tick_count/((double)1000))<<std::endl;
    //std::cout<<"dot(vec_a,vec_b): "<<dot_ab_dev<<std::endl;





}