
// Copyright (c) 2016-17 Karnajit Sen
// University of Erlangen-Nürnberg
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include <chrono>
#include "CUDAPtrchase.h"
#include "cuda_profiler_api.h"
#include <sys/mman.h>
#include<nvml.h>
#include "repeat.h"
#define TBSIZE 1024
__device__  int startIndex = 0;
__shared__ int smv;
void check_error_lat(int temp)
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "Error   : " << temp << " "<< cudaGetErrorString(err) << std::endl;
    exit(err);
  }
}


template <class T>
CUDAPtrchase<T>::CUDAPtrchase(const unsigned int elem, const int strd, const int itr, int pu_ker, int puid_ker ,vector<int> pu_mem,vector<int> puid_mem)
{

  // The array size must be divisible by TBSIZE for kernel launches

  // Set device
  int count;
  cudaGetDeviceCount(&count);
  check_error_lat(0);
  if (pu_ker == 1 && puid_ker >= count)
    throw std::runtime_error("Invalid device index");
  if(pu_ker == 1)
  cudaSetDevice(puid_ker);
  check_error_lat(0);

  // Print out device information

  chase_elem = elem;
  stride = strd;
  iterations = itr;
  puker = pu_ker;
  puidker = puid_ker;
  pumem = pu_mem;
  puidmem = puid_mem;
  

  int temp = puid_mem.size();

  int start= 0;
  int chunk = floor(chase_elem / temp);
  for(int i=1; i<=temp; i++)
   {
      indexchunk.push_back(start);
      start = start + chunk;
      if(i==temp)
           indexchunk.push_back(chase_elem-1);
      else
           indexchunk.push_back(start-1);
   }


  // Check buffers fit on the device
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  if (props.totalGlobalMem < stride*chase_elem*sizeof(T)/4)
    throw std::runtime_error("Device does not have enough memory for all buffers");
}


template <class T>
CUDAPtrchase<T>::~CUDAPtrchase()
{
   
  cout << ".................. " << std::endl;
}

template <class T>
void CUDAPtrchase<T>::freeMemUM()
{
  cudaFree(h_a);
  check_error_lat(0);
}

template <class T>
void CUDAPtrchase<T>::freeMemNonUM()
{
        if(pumem.at(0) == 0)
        {
        cudaFreeHost(h_a);
        check_error_lat(106);
        }
        if(pumem.at(0) == 1)
        {
        cudaFree(h_a);
        check_error_lat(111);
        }
}

template <class T>
void CUDAPtrchase<T>::freeCudaMem()
{
  free(h_a);
  free(h_b);
  cudaFree(d_a);
  check_error_lat(0);
  cudaFree(d_b);
  check_error_lat(0);
}

template <class T>
void CUDAPtrchase<T>::enablePeerAccess()
{
  int count,canAccess;
  cudaGetDeviceCount(&count);
  for ( int i = 0 ; i < count ; i++)
   { 
      cudaSetDevice(i);
     for (int j= 0 ; j < count ; j++ )
      {
       cudaDeviceCanAccessPeer(&canAccess, i,j);
       if(canAccess == 0 || j == i)
       continue;
       cudaDeviceEnablePeerAccess(j,0);
       
       check_error_lat(999);
      }     
   }
  if(puker == 1)
  cudaSetDevice(puidker);
}


template <class T>
void CUDAPtrchase<T>::disablePeerAccess()
{
  int count,canAccess;
  cudaGetDeviceCount(&count);
  for ( int i = 0 ; i < count ; i++)
   {
      cudaSetDevice(i);
     for (int j= 0 ; j < count ; j++ )
      {
       cudaDeviceCanAccessPeer(&canAccess, i,j);
       if(canAccess == 0)
       continue;
       cudaDeviceDisablePeerAccess(j);
       check_error_lat(1000);
      }
   }
  if(puker == 1)
  cudaSetDevice(puidker);
}

template <class T>
void CUDAPtrchase<T>::allocate_arrays_um()
{
 
    	cudaMallocManaged((void **) &h_a, stride*(chase_elem) * sizeof(T));
	check_error_lat(0);   
   	cudaMalloc(&d_time, sizeof(float));
     if(puker == 1)
     {
  	cudaMalloc(&xj, sizeof(T));
     	cudaMalloc(&xi, sizeof(T));
     }
    else
    {
     xj = (T*) malloc(sizeof(T));
     xi = (T*) malloc(sizeof(T));
    }
}


template <class T>
void CUDAPtrchase<T>::allocate_arrays_nonum()
{
          
	if(pumem.at(0) == 0)
        {	
        cudaMallocHost((void **) &h_a, stride * chase_elem * sizeof(T));
        h_b = (T *) malloc(stride * chase_elem * sizeof(T));
  	check_error_lat(106);
  	}
        if(pumem.at(0) == 1)
        {
        cudaSetDevice(puidmem.at(0));
        cudaMalloc((void **) &h_a, stride * chase_elem * sizeof(T));
  	check_error_lat(111);  	
        }       
      
     cudaSetDevice(puidker);
     
  	cudaMalloc(&d_time, sizeof(float));
   
       if(puker == 1)
     {
        cudaMalloc(&xj, sizeof(T));
        cudaMalloc(&xi, sizeof(T));
     }
    else
    {
     xj = (T*) malloc(sizeof(T));
     xi = (T*) malloc(sizeof(T));
    }

}

template <class T>
void CUDAPtrchase<T>::allocate_arrays_device_lat()
{
    	
  	cudaMallocHost((void **) &h_a, stride*chase_elem * sizeof(T));
  	check_error_lat(1);
  	 
  	cudaMalloc(&d_a, stride*chase_elem*sizeof(T));
  	check_error_lat(0);

 	cudaMalloc(&d_time, sizeof(unsigned long long));
  	cudaMalloc(&xj, sizeof(T));
 	cudaMalloc(&xi, sizeof(T));
}

template <typename T>
__global__ void init_kernel(T * a, int elements, int stride)
{
	for(unsigned int i=0; i < elements*stride ; i++)
      {
         a[i] =  (i + stride) % (elements*stride); 
      }
}

template <class T>
void CUDAPtrchase<T>::init_arrays_um()
{

     if(pumem.at(0) == 1)
        {
        cudaSetDevice(puidmem.at(0));
        init_kernel<<<1,1>>>(h_a,chase_elem,stride);
        cudaDeviceSynchronize();
         cudaSetDevice(puidker);
        }
      else{
      for(unsigned int i=0; i < chase_elem*stride ; i++)
      {
         h_a[i] =  (i + stride) % (chase_elem*stride);
      }
     }


}


template <class T>
void CUDAPtrchase<T>::init_arrays_num()
{
         if(pumem.at(0) == 1)
        {
        cudaSetDevice(puidmem.at(0));
	init_kernel<<<1,1>>>(h_a,chase_elem,stride);
        cudaDeviceSynchronize();
         cudaSetDevice(puidker);
        }      
      else{
      for(unsigned int i=0; i < chase_elem*stride ; i++)
      {
         h_b[i] =  (i + stride) % (chase_elem*stride); 
      }
    cudaMemcpy(h_a, h_b, stride*chase_elem * sizeof(T), cudaMemcpyHostToHost);
     }
     
}


template <class T>
void CUDAPtrchase<T>::write_arrays()
{
   // Copy host memory to device
  cudaMemcpy(d_a, h_a , chase_elem*stride*sizeof(T), cudaMemcpyHostToDevice); 
  check_error_lat(0);
}

template <class T>
void touchCPU(T* a, int startpoint, int endpoint, int stride)
{

 for (int i = startpoint; i<= endpoint; i++)
  smv = a[i*stride];
}

template <class T>
__global__ void touchGPU(T *a, int startpoint, int endpoint, int stride)
{
   for (int i = startpoint; i<= endpoint; i++)
   smv = a[i*stride];

}

template <class T>
void CUDAPtrchase<T>::distributeUMemory()
{
 for(int i= 0, j= 0; i < pumem.size(); i++, j+=2)
  {
    if(pumem.at(i)== 0)
    {
      touchCPU(h_a,indexchunk.at(j),indexchunk.at(j+1),stride);
    }
     if(pumem.at(i)== 1)
    {
      cudaSetDevice(puidmem.at(i));
      touchGPU<<<1,1>>>(h_a,indexchunk.at(j),indexchunk.at(j+1),stride);
      cudaDeviceSynchronize();
    }
  }
  cudaSetDevice(puidker);

}

template <typename T>
void latency_cpu(T *A, int N, int iterations,  T *xj, T *xi)
{

   T j = startIndex;

    #pragma omp parallel for
    for (int it=0; it < N*iterations; it++)
    {
       j=A[j];
    }

     *xj = j;
}


template <typename T>
__global__ void latency_kernel(T *A, int N, int iterations,  T *xj, T *xi)
{

   T j = startIndex;  
 
 
    for (int it=0; it < N*iterations; it++) 
    {
       j=A[j];
    }

     *xj = j;
}

template <class T>
double CUDAPtrchase<T>::latency()
{
  std::chrono::high_resolution_clock::time_point t1, t2;

   if(puker == 1)
 {
   cudaProfilerStart();
   t1 = std::chrono::high_resolution_clock::now(); 
  latency_kernel<<<1,1>>>(h_a, chase_elem, iterations, xj, xi);
  cudaDeviceSynchronize();
   t2 = std::chrono::high_resolution_clock::now();
   h_time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  cudaProfilerStop();
  check_error_lat(12);
  }
  if (puker == 0)
  {
   t1 = std::chrono::high_resolution_clock::now();
  latency_cpu(h_a, chase_elem, iterations, xj, xi);
   t2 = std::chrono::high_resolution_clock::now();
   h_time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

  }
  return h_time;
}

template class CUDAPtrchase<int>;
