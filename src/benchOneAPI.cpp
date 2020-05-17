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

#define VECType float

int main() {
  
  
  // declare variables in the host memory
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
  std::cout<<"             Computing vector dot product on the host                            "<<std::endl;
  std::cout<<"================================================================================="<<std::endl;

  for (int i = 0; i<SIZE; ++i)
    dot_ab_host = dot_ab_host + (vec_a[i]*vec_b[i]);
  
  std::cout<<"dot(vec_a,vec_b): "<<dot_ab_host<<std::endl;

  std::cout<<"================================================================================="<<std::endl;
  std::cout<<"             Computing vector dot product on the gpu_selctor                     "<<std::endl;
  std::cout<<"================================================================================="<<std::endl;


  // mannual reduction without using any libraries. 
  {

    sycl::default_selector gpu;
    sycl::queue d_queue(gpu);

    auto a_size = sycl::range<1>{vec_a.size()};
    auto b_size = sycl::range<1>{vec_b.size()};

    auto dot_ab_size = sycl::range<1>{1};

    sycl::buffer<VECType, 1>  _vec_a(vec_a.data(), a_size);
    sycl::buffer<VECType, 1>  _vec_b(vec_b.data(), b_size);
    sycl::buffer<VECType, 1>  _dot_ab_dev(&dot_ab_dev, dot_ab_size);

    auto wgroup_size = 32;
    auto n_wgroups = (a_size/wgroup_size);
    sycl::buffer<VECType, 1> _g_mem(sycl::range<1>{n_wgroups});

    d_queue.submit([&](sycl::handler &cgh) {
        
        auto dot_ab = _dot_ab_dev.get_access<sycl::access::mode::write>(cgh);
        auto a_in   = _vec_a.get_access<sycl::access::mode::read>(cgh);
        auto b_in   = _vec_b.get_access<sycl::access::mode::read>(cgh);
        auto g_in   = _g_mem.get_access<sycl::access::mode::write>(cgh);

        sycl::accessor <VECType, 1, sycl::access::mode::read_write, sycl::access::target::local> local_mem(sycl::range<1>(wgroup_size), cgh);
        sycl::stream out(1024, 256, cgh);
        

        cgh.parallel_for<class reduction_kernel>(sycl::nd_range<1>(n_wgroups * wgroup_size, wgroup_size), [=] (sycl::nd_item<1> item) {
          //out<<"num:groups: "<<n_wgroups<<"\n";
          size_t local_id  = item.get_local_linear_id();
          size_t global_id = item.get_global_linear_id();
          size_t group_id  = item.get_group_linear_id();
          
          local_mem[local_id] = 0.0;
          
          if ((global_id) < a_in.get_count()) {
              local_mem[local_id] = (a_in[global_id] * b_in[global_id]);
              //out<<"local id : "<<local_id<<"val: "<<local_mem[local_id]<<sycl::endl;
          }


          // reduction on the local mem. 
          item.barrier(sycl::access::fence_space::local_space);
          for (size_t stride = 1; stride < wgroup_size; stride *= 2) {
              auto idx = 2 * stride * local_id;
              if (idx < wgroup_size) {
                local_mem[idx] = local_mem[idx] + local_mem[idx + stride];
                
              }

              item.barrier(sycl::access::fence_space::local_space);
          }

          if(local_id==0)
            g_in[group_id] = local_mem[0];

          
        });

    });


    d_queue.submit([&](sycl::handler &cgh) {

      auto dot_ab = _dot_ab_dev.get_access<sycl::access::mode::write>(cgh);
      // auto a_in   = _vec_a.get_access<sycl::access::mode::read>(cgh);
      // auto b_in   = _vec_b.get_access<sycl::access::mode::read>(cgh);
      auto g_in   = _g_mem.get_access<sycl::access::mode::write>(cgh);
    
      cgh.single_task<class vec_dot>([=]() {
        for(unsigned int idx =0; idx < g_in.get_count(); idx++)
          dot_ab[0] = dot_ab[0] + g_in[idx];
      });
    
    });




  }

  std::cout<<"dot(vec_a,vec_b): "<<dot_ab_dev<<std::endl;

  return 0;

}
