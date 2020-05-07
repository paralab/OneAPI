/**
 * @file benchOneAPI.h
 * @brief simple benchmark performing, 
 *  1. vector dot product
 *  2. Matrix vector product
 *  3. Matrix matrix product.  
 * @version 0.1
 * @date 2020-05-01
 * @copyright Copyright (c) 2020
 * 
 */
#include <vector>
#include <iostream>
#include <CL/sycl.hpp>

#define SIZE 1024
namespace sycl = cl::sycl;


int main() {
  
  
  // declare variables in the host memory
  std::array<double, SIZE> vec_a;
  std::array<double, SIZE> vec_b;

  float dot_ab_dev=0;
  double dot_ab_host=0;

  for (int i = 0; i<SIZE; ++i)
  {
    vec_a[i] = 1.0;
    vec_b[i] = 1.0;
  }


  std::cout<<"================================================================================="<<std::endl;
  std::cout<<"             Computing vector dot product on the host                            "<<std::endl;
  std::cout<<"================================================================================="<<std::endl;

  for (int i = 0; i<SIZE; ++i)
    dot_ab_host = dot_ab_host + (vec_a[i]*vec_b[i]);
  
  std::cout<<"dot(vec_a,vec_b): "<<dot_ab_host<<std::endl;

  std::cout<<"================================================================================="<<std::endl;
  std::cout<<"             Computing vector dot product on the gpu_selctor                     "<<std::endl;
  std::cout<<"================================================================================="<<std::endl;

  {

    sycl::gpu_selector gpu;
    sycl::queue d_queue(gpu);

    auto a_size = sycl::range<1>{vec_a.size()};
    auto b_size = sycl::range<1>{vec_b.size()};

    auto dot_ab_size = sycl::range<1>{1};

    sycl::buffer<double, 1>  _vec_a(vec_a.data(), a_size);
    sycl::buffer<double, 1>  _vec_b(vec_b.data(), b_size);
    sycl::buffer<float, 1>  _dot_ab_dev(&dot_ab_dev, dot_ab_size);
    
    d_queue.submit([&](sycl::handler &cgh) {
        
        auto dot_ab = _dot_ab_dev.get_access<sycl::access::mode::read_write>(cgh);
        auto a_in   = _vec_a.get_access<sycl::access::mode::read>(cgh);
        auto b_in   = _vec_b.get_access<sycl::access::mode::read>(cgh);

        dot_ab[0]=0;
        
        cgh.parallel_for<class vec_dot>(a_size,[=](sycl::id<1> idx) {
            dot_ab[0] = dot_ab[0] + (a_in[idx] * b_in[idx]);
        });

    });


  }

  std::cout<<"dot(vec_a,vec_b): "<<dot_ab_dev<<std::endl;

  return 0;

}
