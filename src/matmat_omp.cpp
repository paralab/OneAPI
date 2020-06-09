#include<iostream>
#include <chrono>
#include <ctime>
#include<assert.h>

int main(int argc, char** argv) 
{
    if(argc < 2)
    {
      std::cout<<"Usage: "<<argv[0]<<" N iter"<<std::endl;
      exit(0);
    }

    unsigned int N = atoi(argv[1]);
    unsigned int iter = atoi(argv[2]);
    
    double* mat1 = new double[N*N];
    double* mat2 = new double[N*N];
    double* rmat = new double[N*N];

    for(unsigned int i=0; i < N; i ++)
     for(unsigned int j=0; j < N; j++)
      mat1[i*N+j]=1.0;
    
    for(unsigned int i=0; i < N; i ++)
     for(unsigned int j=0; j < N; j++)
      mat2[i*N+j]=1.0;

    auto t_start = std::chrono::high_resolution_clock::now();
    
    #pragma novector noparallel nounroll
    for(unsigned int it=0; it < iter; it++)
    {
        for(unsigned int i=0; i < N; i ++)
        for(unsigned int j=0; j < N; j++)
        {  
            rmat[i*N + j] = 0.0;
            for(unsigned int w=0; w < N; w++)
                rmat[i*N + j] += (mat1[i*N +w] * mat2[w*N +j]);
        }
    }

    // for(unsigned int i=0; i < N; i ++)
    //  for(unsigned int j=0; j < N; j++)
    //  {
    //     std::cout<<"rmat:"<<rmat[i*N+j]<<std::endl;
    //  }

    auto t_end = std::chrono::high_resolution_clock::now();
    double tick_count=(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
    std::cout<<"time host(loop 1) (s): "<<(tick_count/((double)1000))<<std::endl;
      

    t_start = std::chrono::high_resolution_clock::now();
    #pragma novector noparallel nounroll
    for(unsigned int it=0; it < iter; it++)
    {
        const unsigned int s=16;
        
        for(unsigned int w=0; w < N*N;w++)
            rmat[w]=0.0;

        for(int jj=0; jj<N; jj+= s){
            for(int kk=0; kk<N; kk+= s){
                for(int i=0;i<N;i++){
                    for(int j = jj; j<((jj+s)>N?N:(jj+s)); j++){
                        double temp = 0;
                        for(int k = kk; k<((kk+s)>N?N:(kk+s)); k++){
                            temp += mat1[i*N+k]*mat2[k*N+j];
                        }
                        rmat[i*N+j]+= temp;
                    }
                }
            }
        }
        
    }

    // for(unsigned int i=0; i < N; i ++)
    //  for(unsigned int j=0; j < N; j++)
    //  {
    //     std::cout<<"rmat:"<<rmat[i*N+j]<<std::endl;
    //  }
    

    t_end = std::chrono::high_resolution_clock::now();
    tick_count=(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
    std::cout<<"time host(loop 2) (s): "<<(tick_count/((double)1000))<<std::endl;


    // t_start = std::chrono::high_resolution_clock::now();
    // #pragma novector noparallel nounroll
    // for(unsigned int it=0; it < iter; it++)
    // {
    //     const unsigned int s=16;
    //     for(int jj=0; jj<N; jj+= s){
    //         for(int kk=0; kk<N; kk+= s){
    //             for(int i=0;i<N;i++){
    //                 for(int j = jj; j<((jj+s)>N?N:(jj+s)); j++){
    //                     double temp = 0;
    //                     #pragma omp simd
    //                     for(int k = kk; k<((kk+s)>N?N:(kk+s)); k++){
    //                         temp += mat1[i*N+k]*mat2[k*N+j];
    //                     }
    //                     rmat[i*N+j]+= temp;
    //                 }
    //             }
    //         }
    //     }
        
    // }

    // t_end = std::chrono::high_resolution_clock::now();
    // tick_count=(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
    // std::cout<<"time host(loop 3) (s): "<<(tick_count/((double)1000))<<std::endl;



    delete [] mat1;
    delete [] mat2;
    delete [] rmat;


}