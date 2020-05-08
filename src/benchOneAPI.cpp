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

  double dot_ab_dev=0;
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
    sycl::buffer<double, 1>  _dot_ab_dev(&dot_ab_dev, dot_ab_size);
    
   
    // d_queue.submit([&](sycl::handler &cgh) {
        
    //     auto dot_ab = _dot_ab_dev.get_access<sycl::access::mode::read_write>(cgh);
    //     auto a_in   = _vec_a.get_access<sycl::access::mode::read>(cgh);
    //     auto b_in   = _vec_b.get_access<sycl::access::mode::read>(cgh);

    //     cgh.single_task<class vec_dot>([=]() {
    //         for(unsigned int idx =0; idx < a_in.get_count(); idx++)
    //           dot_ab[0] = dot_ab[0] + (a_in[idx] * b_in[idx]);
    //     });

    // });


    d_queue.submit([&](sycl::handler &cgh) {
        
        auto dot_ab = _dot_ab_dev.get_access<sycl::access::mode::write>(cgh);
        auto a_in   = _vec_a.get_access<sycl::access::mode::read>(cgh);
        auto b_in   = _vec_b.get_access<sycl::access::mode::read>(cgh);

        auto wgroup_size = 32;
        auto n_wgroups = (a_in.get_count()/wgroup_size);

        sycl::accessor <double, 1, sycl::access::mode::read_write, sycl::access::target::local> local_mem(sycl::range<1>(wgroup_size), cgh);
        sycl::accessor <double, 1, sycl::access::mode::read_write, sycl::access::target::local> group_mem(sycl::range<1>(n_wgroups), cgh);
        sycl::stream out(1024, 256, cgh);
        

        cgh.parallel_for<class reduction_kernel>(sycl::nd_range<1>(n_wgroups * wgroup_size, wgroup_size), [=] (sycl::nd_item<1> item) {
          //out<<"num:groups: "<<n_wgroups<<"\n";
          size_t local_id  = item.get_local_linear_id();
          size_t global_id = item.get_global_linear_id();
          size_t group_id  = item.get_group_linear_id();
          
          local_mem[local_id] = 0;

          if ((global_id) < a_in.get_count()) {
              local_mem[local_id] = (a_in[global_id] * b_in[global_id]);
              //out<<"local id : "<<local_id<<"val: "<<local_mem[local_id]<<sycl::endl;
          }

          item.barrier(sycl::access::fence_space::local_space);

          for (size_t stride = 1; stride < wgroup_size; stride *= 2) {
              auto idx = 2 * stride * local_id;
              if (idx < wgroup_size) {
                local_mem[idx] = local_mem[idx] + local_mem[idx + stride];
                
              }

              item.barrier(sycl::access::fence_space::local_space);
          }

          if (local_id == 0) {
            group_mem[group_id] = local_mem[0];
          }
          item.barrier(sycl::access::fence_space::global_space);

          if(local_id==0)
            out<<"group_id: "<<group_id<< ": "<<group_mem[group_id]<<"\n";

          if(local_id==0)
          {

            for (size_t stride = 1; stride < n_wgroups; stride *= 2) {
              auto idx = 2 * stride * group_id;
              if (idx < n_wgroups) {
                group_mem[idx] = group_mem[idx] + group_mem[idx + stride];
                
              }

              //item.barrier(sycl::access::fence_space::global_space);
            }

            if(group_id==0)
              dot_ab[0] = group_mem[0];

          }
          

          
            

        });

    });


    




  }

  std::cout<<"dot(vec_a,vec_b): "<<dot_ab_dev<<std::endl;

  return 0;

}
