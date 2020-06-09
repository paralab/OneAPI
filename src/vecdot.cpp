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

//#define SIZE 1024
namespace sycl = cl::sycl;
#define VECType double

int main(int argc, char** argv) 
{

    if(argc < 2)
    {
      std::cout<<"Usage: "<<argv[0]<<" vecSz iter"<<std::endl;
      exit(0);
    }

    const unsigned int VEC_SIZE = atoi(argv[1]);
    const unsigned int ITER = atoi(argv[2]);
    double tick_count=0;

    std::vector<VECType> vec_a;
    std::vector<VECType> vec_b;

    vec_a.resize(VEC_SIZE);
    vec_b.resize(VEC_SIZE);

    VECType dot_ab_dev  = 0.0;
    VECType dot_ab_host = 0.0;

    for (int i = 0; i<VEC_SIZE; ++i)
    {
        vec_a[i] = 1.0;
        vec_b[i] = 1.0;
    }


    std::cout<<"================================================================================="<<std::endl;
    std::cout<<"             Computing vector dot product on the host(CPU)                       "<<std::endl;
    std::cout<<"================================================================================="<<std::endl;

    auto t_start = std::chrono::high_resolution_clock::now();
    #pragma novector noparallel nounroll
    for(unsigned int t=0; t < ITER; t++)
    {
      //std::cout<<"iter: "<<t<<std::endl;
      dot_ab_host=0;
      #pragma novector noparallel nounroll
      for (int i = 0; i<VEC_SIZE; ++i)
        dot_ab_host = dot_ab_host + (vec_a[i]*vec_b[i]);

    }
    auto t_end = std::chrono::high_resolution_clock::now();
    tick_count=(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
    std::cout<<"time host (s): "<<(tick_count/((double)1000))<<std::endl;
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

        auto a_size = sycl::range<1>{vec_a.size()};
        auto b_size = sycl::range<1>{vec_b.size()};
        auto dot_ab_size = sycl::range<1>{1};

        assert(a_size==b_size);

        sycl::buffer<VECType, 1>  _vec_a(vec_a.data(), a_size);
        sycl::buffer<VECType, 1>  _vec_b(vec_b.data(), b_size);
        sycl::buffer<VECType, 1>  _dot_ab_dev(&dot_ab_dev, dot_ab_size);

        for(unsigned int t=0; t < ITER; t++)
          mkl::blas::dot(d_queue, vec_a.size(), _vec_a, 1, _vec_b, 1, _dot_ab_dev);

        d_queue.wait();


        



    }catch (cl::sycl::exception const &e) 
    {
        std::cout << "\t\tSYCL exception during GEMM\n"
              << e.what() << std::endl
              << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    //std::cout<<"time device kernel execution (s): "<<std::chrono::duration_cast<std::chrono::microseconds>(startTimeList[1]-startTimeList[0]).count()<<std::endl;
    t_end = std::chrono::high_resolution_clock::now();
    tick_count=(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
    std::cout<<"time device (s): "<<(tick_count/((double)1000))<<std::endl;
    std::cout<<"dot(vec_a,vec_b): "<<dot_ab_dev<<std::endl;


   return 0;


}