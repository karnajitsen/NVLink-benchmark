
// Copyright (c) 2016-17 Karnajit Sen
// University of Erlangen-Nürnberg
//
// For full license terms please see the LICENSE file distributed with this
// source code
#include <omp.h>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include "CUDAStream.h"
#include "curand_kernel.h"
#include "cuda_profiler_api.h"
#include <sys/mman.h>
#include<nvml.h>
#include<math.h>
#include <vector>
#define TBSIZE 1024
#define LEN 30
__shared__ double smv,smv1,smv2,smv3,smv4,smv5,smv6;

volatile double t1,t2,t3,t4,t5,t6;
__device__ double j;
__device__ int *aa;

void check_error(int temp)
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "Error   : " << temp << " "<< cudaGetErrorString(err) << std::endl;
    exit(err);
  }
}

template <class T>
CUDAStream<T>::CUDAStream(const unsigned int ARRAY_SIZE, int str,  int pu_ker, int puid_ker ,std::vector<int> pu_mem,std::vector<int> puid_mem)
{

  // The array size must be divisible by TBSIZE for kernel launches
  /*if (ARRAY_SIZE % TBSIZE != 0)
  {
    std::stringstream ss;
    ss << "Array size must be a multiple of " << TBSIZE;
    throw std::runtime_error(ss.str());
  }*/

  // Set device
  int count;
  cudaGetDeviceCount(&count);
 // cout << "device count : " << puid_ker << std::endl;
  check_error(1000);
  if (pu_ker == 1 && puid_ker >= count)
    throw std::runtime_error("Invalid device id");
  if (pu_ker == 1)
  cudaSetDevice(puid_ker);
  check_error(100);
 // cudaMalloc((void **) &aa,LEN * sizeof(int));
  // Print out device information
//  std::cout << "Using CUDA device " << getDeviceName(device_index) << std::endl;
 // std::cout << "Driver: " << getDeviceDriver(device_index) << std::endl;

  array_size = ARRAY_SIZE;
  stride = str;
  puker = pu_ker;
  puidker = puid_ker;
  pumem = pu_mem;
  puidmem = puid_mem;
 
  int temp = puid_mem.size();
 //if(array_size % puid_mem.size() != 0)
   
  int start= 0;
  int chunk = floor(array_size / temp);
  for(int i=1; i<=temp; i++)
   {
      indexchunk.push_back(start);
      start = start + chunk;
      if(i==temp)
           indexchunk.push_back(array_size-1);
      else
           indexchunk.push_back(start-1);
   }
  
  
  // Check buffers fit on the device
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
   if (props.totalGlobalMem < 3*ARRAY_SIZE*sizeof(T))
   throw std::runtime_error("Device does not have enough memory for all 3 buffers");
}


template <class T>
CUDAStream<T>::~CUDAStream()
{
   
  //cudaDeviceReset();
//  cout << ".................. " << std::endl;
}

template <class T>
void CUDAStream<T>::freeMemUM()
{
  //cudaSetDevice(0);
  cudaFree(d_a);
  check_error(10);
  cudaFree(d_b);
  check_error(11);
  cudaFree(d_c);
  check_error(12);
  cudaFree(d_d);
  check_error(13);
  cudaFree(d_e);
  check_error(14);
  cudaFree(d_f);
  check_error(15);
  
}

template <class T>
void CUDAStream<T>::freeMemNonUM()
{
  cudaFreeHost(num_a);
  cudaFreeHost(num_b);
  cudaFreeHost(num_c);
  cudaFreeHost(num_d);
  cudaFreeHost(num_e);
  if(puker == 1)
    cudaFree(d_f);  
   else
    free(d_f);
}

template <class T>
void CUDAStream<T>::freeCudaMem()
{
  check_error(0);
  cudaFree(d_a);
  cudaFree(d_b); 
  check_error(0);
}


template <class T>
void CUDAStream<T>::enablePeerAccess()
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
       
       check_error(999);
      }     
   }
  if(puker == 1)
  cudaSetDevice(puidker);
}


template <class T>
void CUDAStream<T>::disablePeerAccess()
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
       cudaDeviceDisablePeerAccess(j);
       check_error(1000);
      }
   }
  if(puker == 1)
  cudaSetDevice(puidker);
}



template <class T>
void CUDAStream<T>::allocate_arrays_cudamem()
{
     cudaSetDevice(0);
    cudaMallocHost((void **) &h_a, stride*array_size * sizeof(T));
    check_error(1);
    cudaSetDevice(1);
    cudaMalloc(&d_a, stride*array_size*sizeof(T));
    check_error(0);

 
    h_b = (T *) malloc( stride*array_size * sizeof(T));
    
    cudaMalloc(&d_b, stride*array_size*sizeof(T));
    check_error(0);
    
    
 }


template <class T>
void CUDAStream<T>::allocate_arrays_um(const char memAdvise)
{
      //  cudaSetDevice(0); 
    	cudaMallocManaged((void **) &d_a, stride * array_size * sizeof(T));
	 check_error(1010);
  	cudaMallocManaged((void **) &d_b, stride * array_size * sizeof(T));
  	check_error(1020);
  	cudaMallocManaged((void **) &d_c, stride * array_size * sizeof(T));
  	check_error(1030);
  	cudaMallocManaged((void **) &d_d, stride * array_size * sizeof(T));
  	check_error(1040);
	cudaMallocManaged((void **) &d_e, stride * array_size * sizeof(T));
        check_error(1050);

         cudaMallocManaged((void **) &d_f, stride * array_size * sizeof(T));
        check_error(1060);


    if(puker == 1 && memAdvise == 'Y'){
   //  cudaSetDevice(puidker);
  //  cout << "in Mem advise " << std::endl;
    cuMemAdvise ( (CUdeviceptr) d_a, array_size*stride*sizeof(T),CU_MEM_ADVISE_SET_ACCESSED_BY , puidker );
     cuMemAdvise ( (CUdeviceptr) d_b, array_size*stride*sizeof(T),CU_MEM_ADVISE_SET_ACCESSED_BY, puidker );
    cuMemAdvise ( (CUdeviceptr) d_c, array_size*stride*sizeof(T),CU_MEM_ADVISE_SET_ACCESSED_BY , puidker );
     cuMemAdvise ( (CUdeviceptr) d_d, array_size*stride*sizeof(T),CU_MEM_ADVISE_SET_ACCESSED_BY, puidker );
    cuMemAdvise ( (CUdeviceptr) d_e, array_size*stride*sizeof(T),CU_MEM_ADVISE_SET_ACCESSED_BY , puidker );
     cuMemAdvise ( (CUdeviceptr) d_f, array_size*stride*sizeof(T),CU_MEM_ADVISE_SET_ACCESSED_BY, puidker );

  }
  else if(memAdvise == 'Y')
  {
     cuMemAdvise ( (CUdeviceptr) d_a, array_size*stride*sizeof(T),CU_MEM_ADVISE_SET_ACCESSED_BY,CU_DEVICE_CPU);
   cuMemAdvise ( (CUdeviceptr) d_b, array_size*stride*sizeof(T), CU_MEM_ADVISE_SET_ACCESSED_BY,CU_DEVICE_CPU) ;
       cuMemAdvise ( (CUdeviceptr) d_c, array_size*stride*sizeof(T),CU_MEM_ADVISE_SET_ACCESSED_BY,CU_DEVICE_CPU);
   cuMemAdvise ( (CUdeviceptr) d_d, array_size*stride*sizeof(T), CU_MEM_ADVISE_SET_ACCESSED_BY,CU_DEVICE_CPU) ;
        cuMemAdvise ( (CUdeviceptr) d_e, array_size*stride*sizeof(T),CU_MEM_ADVISE_SET_ACCESSED_BY,CU_DEVICE_CPU);
   cuMemAdvise ( (CUdeviceptr) d_f, array_size*stride*sizeof(T), CU_MEM_ADVISE_SET_ACCESSED_BY,CU_DEVICE_CPU) ;
 }

       
 }


template <class T>
void CUDAStream<T>::allocate_arrays_nonum()
{
    	
        h_a = (T *) malloc(stride * array_size * sizeof(T));
  	h_b = (T *) malloc(stride * array_size * sizeof(T));
  	h_c = (T *) malloc(stride * array_size * sizeof(T));
  	h_d = (T *) malloc(stride * array_size * sizeof(T));
  	h_e = (T *) malloc(stride * array_size * sizeof(T));
         //cudaSetDevice(puidker);
        cudaHostAlloc((void **) &num_a, pumem.size()*sizeof(T*),cudaHostAllocMapped);
  	check_error(101);
  	cudaHostAlloc((void **) &num_b, pumem.size()*sizeof(T*),cudaHostAllocMapped);
  	check_error(102);
  	cudaHostAlloc((void **) &num_c, pumem.size()*sizeof(T*),cudaHostAllocMapped);
  	check_error(103);
  	cudaHostAlloc((void **) &num_d, pumem.size()*sizeof(T*),cudaHostAllocMapped);
  	check_error(104);
  	cudaHostAlloc((void **) &num_e, pumem.size()*sizeof(T*),cudaHostAllocMapped);
        check_error(105);
         cudaHostGetDevicePointer(&dnum_a,num_a,0);
         cudaHostGetDevicePointer(&dnum_b, num_b,0);
         cudaHostGetDevicePointer(&dnum_c, num_c,0);
         cudaHostGetDevicePointer(&dnum_d, num_d,0);
         cudaHostGetDevicePointer(&dnum_e, num_e,0);
        
      for(int i = 0; i<puidmem.size() ; i++)
      {
        int bytes = indexchunk.at(2*i+1) - indexchunk.at(2*i)+1;
        if(pumem.at(i) == 0)
        {	
        cudaMallocHost((void **) &num_a[i], stride * bytes * sizeof(T));
  	check_error(106);
  	cudaMallocHost((void **) &num_b[i], stride * bytes * sizeof(T));
  	check_error(107);
  	cudaMallocHost((void **) &num_c[i], stride * bytes * sizeof(T));
  	check_error(108);
  	cudaMallocHost((void **) &num_d[i], stride * bytes * sizeof(T));
  	check_error(109);
  	cudaMallocHost((void **) &num_e[i], stride * bytes * sizeof(T));
        check_error(110);
        }
        if(pumem.at(i) == 1)
        {
         cudaSetDevice(puidmem.at(0));
        cudaMalloc((void **) &dnum_a[i], stride * bytes * sizeof(T));
  	check_error(111);
  	cudaMalloc((void **) &dnum_b[i], stride * bytes * sizeof(T));
  	check_error(112);
  	cudaMalloc((void **) &dnum_c[i], stride * bytes * sizeof(T));
  	check_error(113);
  	cudaMalloc((void **) &dnum_d[i], stride * bytes * sizeof(T));
  	check_error(114);
  	cudaMalloc((void **) &dnum_e[i], stride * bytes * sizeof(T));
        check_error(115);

        }       
      }
     if(puker == 1)
     {
     cudaSetDevice(puidker);
     cudaMalloc((void **) &d_f, stride * array_size * sizeof(T));
     check_error(5);
     }
     else
     cudaMallocHost((void **) &d_f, stride * array_size * sizeof(T));
  	
}

template <class T>
void CUDAStream<T>::allocate_arrays_membw()
{
    	
  	cudaMallocHost((void **) &h_a, stride* array_size * sizeof(T));
  	check_error(1);
  	cudaMallocHost((void **) &h_b, stride * array_size * sizeof(T));
  	check_error(2);
  	cudaMallocHost((void **) &h_c, stride * array_size * sizeof(T));
  	check_error(3);
  	cudaMallocHost((void **) &h_d, stride * array_size * sizeof(T));
  	check_error(4);
        cudaMallocHost((void **) &h_e, stride * array_size * sizeof(T));
        check_error(3);
        cudaMallocHost((void **) &h_f, stride * array_size * sizeof(T));
        check_error(4);
 
  	cudaMalloc(&d_a, stride * array_size * sizeof(T));
  	check_error(0);
  	cudaMalloc(&d_b, stride * array_size * sizeof(T));
  	check_error(0);
  	cudaMalloc(&d_c, stride * array_size * sizeof(T));
  	check_error(0);
  	cudaMalloc(&d_d, stride * array_size * sizeof(T));
  	check_error(0);
        cudaMalloc(&d_e, stride * array_size * sizeof(T));
        check_error(0);
        cudaMalloc(&d_f, stride * array_size * sizeof(T));
        check_error(0);

}

template <class T>
void CUDAStream<T>::init_arrays_um()
{
   for (int i = 0; i < array_size*stride ; i++)
   {
      d_a[i] = startA;
      d_b[i] = startB;
      d_c[i] = startC;
      d_d[i] = startD; 
      d_e[i] = startE;     
      //d_f[i] = startD;
   }
  
}


template <class T>
void CUDAStream<T>::init_arrays_num()
{
   for (int i = 0; i < array_size*stride ; i++)
   {
      h_a[i] = startA;
      h_b[i] = startB;
      h_c[i] = startC;
      h_d[i] = startD;
      h_e[i] = startE;
      //d_f[i] = startD;
   }

}

template <class T>
void touchCPU(T* a,  T* b, T* c, T* d, T* e, T* f, int startpoint, int endpoint, int stride)
{
  
 for (int i = startpoint; i<= endpoint; i++)
 {
  t1 = a[i*stride]; 
  t2 = b[i*stride];
  t3 = c[i*stride];
  t4 = d[i*stride];
  t5 = e[i*stride];
  t6 = f[i*stride];
 }
}

template <class T>
__global__ void touchGPU(T *a, T* b, T* c, T* d, T* e, T* f, int startpoint, int endpoint, int stride)
{
   const int i = (blockDim.x * blockIdx.x + threadIdx.x+startpoint)*stride;
    smv1 = a[i];
    smv2 = b[i];
    smv3 = c[i];
    smv4 = d[i];
    smv5 = e[i];
    smv6 = f[i];

}

template <class T>
void CUDAStream<T>::distributeUMemory()
{ 
  //cout << "start dist memory " << std::endl;
 for(int i= 0, j= 0; i < pumem.size(); i++, j+=2)
  {
    if(pumem.at(i)== 0)
    {
      touchCPU(d_a,d_b, d_c, d_d, d_e, d_f, indexchunk.at(j),indexchunk.at(j+1),stride);
    }
     if(pumem.at(i)== 1)
    {
       //cout << "start Distribue mem " << std::endl;
      int d = indexchunk.at(j+1)-indexchunk.at(j);
      int gs = ceil(d/1024.0);      
      cudaSetDevice(puidmem.at(i));
      touchGPU<<<gs,1024>>>(d_a,d_b,d_c,d_d,d_e,d_f,indexchunk.at(j),indexchunk.at(j+1),stride);
      cudaDeviceSynchronize();
      check_error(200);
    }
  }
   if(puker == 1){
  cudaSetDevice(puidker);
 } 
}

template <class T>
void CUDAStream<T>::lockMem()
{
  cout << mlockall(MCL_CURRENT) << std::endl;

}

template <class T>
void CUDAStream<T>::setApplictionClock()
 {
	nvmlReturn_t nvmlError = nvmlInit();
	if (NVML_SUCCESS != nvmlError )
    	fprintf (stderr, "NVML_ERROR: %s (%d) \n", 
             nvmlErrorString( nvmlError ), nvmlError);
	
	//0. Get active CUDA device
	int activeCUDAdevice = 0;
	cudaGetDevice ( &activeCUDAdevice );

	//1. Get device properties of active CUDA device
	cudaDeviceProp activeCUDAdeviceProp;
	cudaGetDeviceProperties ( &activeCUDAdeviceProp, activeCUDAdevice );

	//2. Get number of NVML devices
	unsigned int nvmlDeviceCount = 0;
	nvmlDeviceGetCount ( &nvmlDeviceCount );

	nvmlDevice_t nvmlDeviceId;
	//3. Loop over all NVML devices
	for ( unsigned int nvmlDeviceIdx = 0; nvmlDeviceIdx < nvmlDeviceCount; ++nvmlDeviceIdx )
	{
   	 //4. Obtain NVML device Id
    		nvmlDeviceGetHandleByIndex ( nvmlDeviceIdx, &nvmlDeviceId );
    
   	 //5. Query PCIe Info of the NVML device
   		 nvmlPciInfo_t nvmPCIInfo;
   		 nvmlDeviceGetPciInfo ( nvmlDeviceId, &nvmPCIInfo );
    
   	 //6. Compare NVML device PCI-E info with CUDA device properties
   		 if ( static_cast<unsigned int>(activeCUDAdeviceProp.pciBusID)== nvmPCIInfo.bus && static_cast<unsigned int>(activeCUDAdeviceProp.pciDeviceID)== nvmPCIInfo.device && 
         		static_cast<unsigned int>(activeCUDAdeviceProp.pciDomainID)== nvmPCIInfo.domain )
       			 break;
	}

	//Query current application clock setting
	unsigned int appSMclock = 0;
	unsigned int appMemclock = 0;
	nvmlDeviceGetApplicationsClock ( nvmlDeviceId, NVML_CLOCK_SM, &appSMclock );
	nvmlDeviceGetApplicationsClock ( nvmlDeviceId,NVML_CLOCK_MEM,&appMemclock );

	//Query maximum application clock setting
	unsigned int maxSMclock = 0;
	unsigned int maxMemclock = 0;
	nvmlDeviceGetMaxClockInfo ( nvmlDeviceId, NVML_CLOCK_SM, &maxSMclock );
	nvmlDeviceGetMaxClockInfo ( nvmlDeviceId, NVML_CLOCK_MEM,&maxMemclock );
	
	nvmlEnableState_t isRestricted;
	nvmlDeviceGetAPIRestriction ( nvmlDeviceId, NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS, &isRestricted );
	if ( NVML_FEATURE_DISABLED != isRestricted )
	{
   		cout << "Not allowed!!" << std::endl; 
 		cout << nvmlDeviceSetApplicationsClocks( nvmlDeviceId, maxMemclock, maxSMclock  ) << std::endl;
		
	}
	
	nvmlDeviceResetApplicationsClocks ( nvmlDeviceId );
	nvmlShutdown();

 
 }


template <class T>
void CUDAStream<T>::cudaMemCopyPinned()
{
  cudaMemcpy(d_a, h_a , stride * array_size*sizeof(T), cudaMemcpyHostToDevice); 
  check_error(11); 
}

template <class T>
void CUDAStream<T>::cudaMemCopyNonPinned()
{
  cudaMemcpy(d_b, h_b , stride * array_size*sizeof(T), cudaMemcpyHostToDevice); 
  check_error(10);
  
}



template <class T>
void CUDAStream<T>::distributeNUMemory()
{
  
   for(int i = 0; i<puidmem.size() ; i++)
      {
    int bytes = indexchunk.at(2*i+1) - indexchunk.at(2*i)+1;
      if(pumem.at(i) == 1)
  {
  cudaSetDevice(puidmem.at(i));
  // Copy host memory to device
  cudaMemcpy(dnum_a[i], &h_a[indexchunk.at(2*i)] , stride * bytes * sizeof(T), cudaMemcpyHostToDevice); 
  check_error(500);
  cudaMemcpy(dnum_b[i], &h_b[indexchunk.at(2*i)] , stride * bytes * sizeof(T), cudaMemcpyHostToDevice);
  check_error(501);
  cudaMemcpy(dnum_c[i], &h_c[indexchunk.at(2*i)] , stride * bytes * sizeof(T), cudaMemcpyHostToDevice);
  check_error(502);
  cudaMemcpy(dnum_d[i], &h_d[indexchunk.at(2*i)] , stride * bytes * sizeof(T), cudaMemcpyHostToDevice);
  check_error(503);
  cudaMemcpy(dnum_e[i], &h_e[indexchunk.at(2*i)] , stride * bytes * sizeof(T), cudaMemcpyHostToDevice);
  check_error(504);
  cudaSetDevice(puidker);
  }
  else
   {
  cudaMemcpy(num_a[i], &h_a[indexchunk.at(2*i)] , stride * bytes * sizeof(T), cudaMemcpyHostToHost); 
  check_error(600);
  cudaMemcpy(num_b[i], &h_b[indexchunk.at(2*i)] , stride * bytes * sizeof(T), cudaMemcpyHostToHost);
  check_error(610);
  cudaMemcpy(num_c[i], &h_c[indexchunk.at(2*i)] , stride * bytes * sizeof(T), cudaMemcpyHostToHost);
  check_error(620);
  cudaMemcpy(num_d[i], &h_d[indexchunk.at(2*i)] , stride * bytes * sizeof(T), cudaMemcpyHostToHost);
  check_error(630);
  cudaMemcpy(num_e[i], &h_e[indexchunk.at(2*i)] , stride * bytes * sizeof(T), cudaMemcpyHostToHost);
  check_error(640);
    
   }
 }
}

template <class T>
void CUDAStream<T>::read_arrays_num()
{
  for(int i = 0; i<puidmem.size() ; i++)
      {
    int bytes = indexchunk.at(2*i+1) - indexchunk.at(2*i)+1;
      if(pumem.at(i) == 1)
  {
  cudaSetDevice(puidmem.at(i));
  // Copy host memory to device
  cudaMemcpy( &h_a[indexchunk.at(i)] , num_a[i], stride * bytes * sizeof(T), cudaMemcpyDeviceToHost); 
  check_error(0);
  cudaMemcpy(&h_b[indexchunk.at(i)] , num_b[i], stride * bytes * sizeof(T), cudaMemcpyDeviceToHost);
  check_error(0);
  cudaMemcpy(&h_c[indexchunk.at(i)] , num_c[i],  stride * bytes * sizeof(T), cudaMemcpyDeviceToHost);
  check_error(0);
  cudaMemcpy(&h_d[indexchunk.at(i)] , num_d[i],  stride * bytes * sizeof(T), cudaMemcpyDeviceToHost);
  check_error(0);
  cudaMemcpy(&h_e[indexchunk.at(i)] , num_e[i], stride * bytes * sizeof(T), cudaMemcpyDeviceToHost);
  check_error(0);
  cudaSetDevice(puidker);
  }
  else
   {
  cudaMemcpy( &h_a[indexchunk.at(i)] , num_a[i], stride * bytes * sizeof(T), cudaMemcpyHostToHost); 
  check_error(0);
  cudaMemcpy(&h_b[indexchunk.at(i)] , num_b[i],  stride * bytes * sizeof(T), cudaMemcpyHostToHost);
  check_error(0);
  cudaMemcpy(&h_c[indexchunk.at(i)] , num_c[i], stride * bytes * sizeof(T), cudaMemcpyHostToHost);
  check_error(0);
  cudaMemcpy(&h_d[indexchunk.at(i)] , num_d[i], stride * bytes * sizeof(T), cudaMemcpyHostToHost);
  check_error(0);
  cudaMemcpy(&h_e[indexchunk.at(i)] , num_e[i], stride * bytes * sizeof(T), cudaMemcpyHostToHost);
  check_error(0);
    
   }
 }
}


template <class T>
void CUDAStream<T>::setOMPParams(int cputhreads)
{
  omp_set_num_threads(cputhreads);  
}

template <typename T>
__global__ void write_kernel( T *__restrict__ a, int elements, int stride, int block)
{
  const int i = (blockDim.x * blockIdx.x + threadIdx.x);
    for(int k = 0; k < block; k++)
   a[i*stride+0] = i+k;
   //a[i*stride+1] = i +1;
}


template <class T>
void CUDAStream<T>::write()
{
   // cudaProfilerStart();
    write_kernel<<<gridSize.at(0)/stride, blockSize>>>(d_a,array_size,stride,stride);
    cudaDeviceSynchronize();
   //cudaProfilerStop();
   check_error(0);
}

template <class T>
void CUDAStream<T>::num_write()
{
    //cudaProfilerStart();
    for(int i = 0 ; i < pumem.size() ; i++)
    {
    int elements = indexchunk.at(2*i+1) - indexchunk.at(2*i);
    write_kernel<<<gridSize.at(i), blockSize>>>(dnum_a[i], elements, stride,threadSize);    
    }
    cudaDeviceSynchronize();
    //cudaProfilerStop();
    check_error(0);

}

template <typename T>
__global__ void read_kernel(const T * __restrict__ a, int elements, int stride, int block)
{
  const int i = (blockDim.x * blockIdx.x + threadIdx.x);
  for(int k = 0; k < block; k++)
  smv = a[i*stride+k];
 // smv = i;
}


template <class T>
void CUDAStream<T>::read()
{
    //cudaProfilerStart();
    read_kernel<<<gridSize.at(0), blockSize>>>(d_a,array_size,stride,threadSize);
    cudaDeviceSynchronize();
  // cudaProfilerStop();
    check_error(0);

}

template <class T>
void CUDAStream<T>::num_read()
{
   // cudaProfilerStart();
    for(int i = 0 ; i < pumem.size() ; i++)
    {
    int elements = indexchunk.at(2*i+1) - indexchunk.at(2*i);
    read_kernel<<<gridSize.at(i), blockSize>>>(dnum_a[i], elements, stride,threadSize);    
    }
    cudaDeviceSynchronize();
   // cudaProfilerStop();
    check_error(0);

}


template <typename T>
__global__ void copy_kernel( const T * __restrict__ a, T * __restrict__ c, int elements, int stride, int block)
{
   const int i = (blockDim.x * blockIdx.x + threadIdx.x);
   for(int k = 0; k < block; k++)
     c[i*stride+k] = a[i*stride+k];
     
}

template <class T>
void CUDAStream<T>::copy()
{
   // cudaProfilerStart();
    copy_kernel<<<gridSize.at(0), blockSize>>>(d_a,d_f, array_size,stride,threadSize);
    cudaDeviceSynchronize();
   // cudaProfilerStop();
    check_error(0);

}

template <class T>
void CUDAStream<T>::num_copy()
{

    for(int i = 0 ; i < pumem.size() ; i++)
    {
     //cudaSetDevice(puidmem.at(i));
  //  cudaProfilerStart();
    int elements = indexchunk.at(2*i+1) - indexchunk.at(2*i);
    copy_kernel<<<gridSize.at(i), blockSize>>>(dnum_a[i], &d_f[indexchunk.at(2*i)], elements, stride,threadSize);   
//    cudaDeviceSynchronize(); 
    }
    cudaDeviceSynchronize();
   // cudaProfilerStop();
  
    check_error(400);

}

template <typename T>
__global__ void mul_kernel(T * __restrict__ a, T * __restrict__ c, int elements, int stride, int block)
{
  const T scalar = startScalar;
  const int i = (blockDim.x * blockIdx.x + threadIdx.x);
  for(int k = 0; k < block; k++)
  c[i*stride+k] = scalar * a[i*stride+k];
}

template <class T>
void CUDAStream<T>::mul()
{

    mul_kernel<<<gridSize.at(0), blockSize>>>(d_a,d_f,array_size,stride,threadSize);
    cudaDeviceSynchronize();
   
    check_error(0);
}


template <class T>
void CUDAStream<T>::num_mul()
{

    for(int i = 0 ; i < pumem.size() ; i++)
    {
    int elements = indexchunk.at(2*i+1) - indexchunk.at(2*i);
    mul_kernel<<<gridSize.at(i),  blockSize>>>(num_a[i], &d_f[indexchunk.at(i)], elements, stride,threadSize);    
    }
    cudaDeviceSynchronize();
   
    check_error(0);

}

template <typename T>
__global__ void add_kernel(T * __restrict__ a, T * __restrict__ b, T * __restrict__ c, int elements, int stride, int block)
{
   const int i = (blockDim.x * blockIdx.x + threadIdx.x);
   for(int k = 0; k < block; k++)
   c[i*stride+k] =  a[i*stride+k] + b[i*stride+k];
}


template <class T>
void CUDAStream<T>::add()
{

      add_kernel<<<gridSize.at(0), blockSize>>>(d_a, d_b, d_f, array_size, stride,threadSize);
    cudaDeviceSynchronize();
   
    check_error(0);
}

template <class T>
void CUDAStream<T>::num_add()
{

    for(int i = 0 ; i < pumem.size() ; i++)
    {
    int elements = indexchunk.at(2*i+1) - indexchunk.at(2*i);
    add_kernel<<<gridSize.at(i), blockSize>>>(num_a[i], num_b[i], &d_f[indexchunk.at(i)], elements, stride,threadSize);    
    }
    cudaDeviceSynchronize();
   
    check_error(0);

}

template <typename T>
__global__ void triad_kernel(const T * __restrict__ a, const T * __restrict__ b, T * __restrict__ c, int elements, int stride, int block)
{
  const T scalar = startScalar;
  const int i =( blockDim.x * blockIdx.x + threadIdx.x);
  for(int k = 0; k < block; k++)
  c[i*stride+k] = a[i*stride+k] + scalar * b[i*stride+k];
}

template <class T>
void CUDAStream<T>::triad()
{
  
      triad_kernel<<<gridSize.at(0), blockSize>>>(d_a, d_b, d_f, array_size,stride,threadSize);
    cudaDeviceSynchronize();
   
    check_error(0);
}


template <class T>
void CUDAStream<T>::num_triad()
{

    for(int i = 0 ; i < pumem.size() ; i++)
    {
    int elements = indexchunk.at(2*i+1) - indexchunk.at(2*i);
    triad_kernel<<<gridSize.at(i),  blockSize>>>(num_a[i], num_b[i], &d_f[indexchunk.at(i)], elements, stride,threadSize);    
    }
    cudaDeviceSynchronize();
   
    check_error(0);

}

template <typename T>
__global__ void quadad_kernel(const T * __restrict__ a, const T * __restrict__ b, const T * __restrict__ c , T * __restrict__ d, int elements, int stride, int block){
  const int i =( blockDim.x * blockIdx.x + threadIdx.x);
  for(int k = 0; k < block; k++)
  d[i*stride+k] = a[i*stride+k] + b[i*stride+k] * c[i*stride+k];
}

template <class T>
void CUDAStream<T>::quadad()
{
  
    quadad_kernel<<<gridSize.at(0), blockSize>>>(d_a, d_b, d_c, d_f, array_size,stride,threadSize);
    cudaDeviceSynchronize();
   
    check_error(0);
}

template <class T>
void CUDAStream<T>::num_quadad()
{

    for(int i = 0 ; i < pumem.size() ; i++)
    {
    int elements = indexchunk.at(2*i+1) - indexchunk.at(2*i);
    quadad_kernel<<<gridSize.at(i),  blockSize>>>(num_a[i], num_b[i], num_c[i], &d_f[indexchunk.at(i)], elements, stride,threadSize);    
    }
    cudaDeviceSynchronize();
   
    check_error(0);

}

template <typename T>
__global__ void pentad_kernel(const T * __restrict__ a, const T * __restrict__ b, const T * __restrict__ c , const T * __restrict__ d, T * __restrict__ e, int elements, int stride, int block)
{
  const int i =( blockDim.x * blockIdx.x + threadIdx.x);
  for(int k = 0; k < block; k++)
  e[i*stride+k] = a[i*stride+k] + b[i*stride+k] * c[i*stride+k] + d[i*stride+k];
}

template <class T>
void CUDAStream<T>::pentad()
{
    pentad_kernel<<<gridSize.at(0), blockSize>>>(d_a, d_b, d_c, d_d, d_f, array_size, stride,threadSize);
    cudaDeviceSynchronize();
   
    check_error(0);
}


template <class T>
void CUDAStream<T>::num_pentad()
{

    for(int i = 0 ; i < pumem.size() ; i++)
    {
    int elements = indexchunk.at(2*i+1) - indexchunk.at(2*i);
    pentad_kernel<<<gridSize.at(i),  blockSize>>>(num_a[i], num_b[i], num_c[i], num_d[i], &d_f[indexchunk.at(i)], elements, stride,threadSize);    
    }
    cudaDeviceSynchronize();
   
    check_error(0);

}

template <typename T>
__global__ void hexad_kernel(const T * __restrict__ a, const T * __restrict__ b, const T * __restrict__ c , const T * __restrict__ d, const T * __restrict__ e, T * __restrict__ f, int elements, int stride, int block)
{
  const int i =( blockDim.x * blockIdx.x + threadIdx.x);
  for(int k = 0; k < block; k++)
  f[i*stride+k] = a[i*stride+k] + b[i*stride+k] * c[i*stride+k] + d[i*stride+k] * e[i*stride+k];
}

template <class T>
void CUDAStream<T>::hexad()
{  

    hexad_kernel<<<gridSize.at(0), blockSize>>>(d_a, d_b, d_c, d_d, d_e,d_f,array_size, stride,threadSize);
    cudaDeviceSynchronize();
   
    check_error(0);
}

template <class T>
void CUDAStream<T>::num_hexad()
{
    for(int i = 0 ; i < pumem.size() ; i++)
    {
    int elements = indexchunk.at(2*i+1) - indexchunk.at(2*i);
    hexad_kernel<<<gridSize.at(i),  blockSize>>>(num_a[i], num_b[i], num_c[i], num_d[i], num_e[i], &d_f[indexchunk.at(i)], elements, stride,threadSize);    
    }
    cudaDeviceSynchronize();   
    check_error(0);
}

template <class T>
void CUDAStream<T>::h_read()
{
  #pragma omp parallel for private(t1)
   for(int i = 0; i < array_size*stride ; i+=stride)
      {
        t1 =  d_a[i];
      }
  #pragma omp barrier
}

template <class T>
void CUDAStream<T>::h_write()
{
  #pragma omp parallel for
   for(int i = 0; i < array_size*stride ; i+=stride)
      d_a[i] = i * 1.0;
  #pragma omp barrier
}



template <class T>
void CUDAStream<T>::h_copy()
{
  #pragma omp parallel for
   for(int i = 0; i < array_size*stride ; i+=stride)
    {
      d_f[i] = d_a[i];
   }
   #pragma omp barrier	
}

template <class T>
void CUDAStream<T>::h_mul()
{
  
   #pragma omp parallel for
   for(int i = 0; i < array_size*stride ; i+=stride)
      d_f[i] = 2.0 * d_a[i];
   
}

template <class T>
void CUDAStream<T>::h_add()
{
  #pragma omp parallel for
   for(int i = 0; i < array_size*stride ; i+=stride)
      d_f[i] = d_a[i] + d_b[i];
}

template <class T>
void CUDAStream<T>::h_triad()
{
  #pragma omp parallel for 
  for(int i = 0; i < array_size*stride ; i+=stride)
      d_f[i] = d_a[i] + 2.0 * d_b[i];
}


template <class T>
void CUDAStream<T>::h_quadad()
{
   #pragma omp parallel for
   for(int i = 0; i < array_size*stride ; i+=stride)
      d_f[i] = d_a[i] + d_c[i] * d_b[i];
}

template <class T>
void CUDAStream<T>::h_pentad()
{
 #pragma omp parallel for
   for(int i = 0; i < array_size*stride ; i+=stride)
      d_f[i] = d_a[i] + d_c[i] * d_b[i] + d_d[i];
}

template <class T>
void CUDAStream<T>::h_hexad()
{
 #pragma omp parallel for
   for(int i = 0; i < array_size*stride ; i+=stride)
      d_f[i] = d_a[i] + d_c[i] * d_b[i] + d_d[i] * d_e[i];
}

template <class T>
void CUDAStream<T>::setDeviceParameterUM(int bs,int thread)
{
         gridSize.clear(); 
         if(thread >= bs)
         blockSize = bs;
         else
         blockSize = thread;
    
	 if(thread%blockSize == 0)
                gridSize.push_back(thread/blockSize);
        else
                gridSize.push_back(thread/blockSize + 1);
      threadSize = array_size/thread;
}

template <class T>
void CUDAStream<T>::setDeviceParameterNUM(int bs,int thread)
{
    gridSize.clear();
    if(thread >= bs)
         blockSize = bs;
         else
         blockSize = thread;

    for(int i = 0; i < pumem.size(); i++)
    {
       int elements = indexchunk.at(i+1) - indexchunk.at(i);
	 if(elements%bs == 0)
                gridSize.push_back(elements/bs);
        else
                gridSize.push_back(elements/bs + 1);
  //    cout << gridSize.at(0) << " " << blockSize << std::endl;
    }
  threadSize = array_size/thread;
}

void listDevices(void)
{
  // Get number of devices
  int count;
  cudaGetDeviceCount(&count);
  check_error(0);

  // Print device names
  if (count == 0)
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << std::endl;
    std::cout << "Devices:" << std::endl;
    for (int i = 0; i < count; i++)
    {
      std::cout << i << ": " << getDeviceName(i) << std::endl;
    }
    std::cout << std::endl;
  }
}


std::string getDeviceName(const int device)
{
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);
  check_error(0);
  return std::string(props.name);
}


std::string getDeviceDriver(const int device)
{
  cudaSetDevice(device);
  check_error(0);
  int driver;
  cudaDriverGetVersion(&driver);
  check_error(0);
  return std::to_string(driver);
}

template class CUDAStream<float>;
template class CUDAStream<double>;
