
// Copyright (c) 2016-17 , Karnajit Sen
// IBM Deutschland R&D GmbH, University of Erlangen-Nürnberg
//
// For full license terms please see the LICENSE file distributed with this
// source code
#include <cuda_profiler_api.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <limits>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <fstream>
#include <array>
#include "common.h"
#include "Stream.h"
#include "parameters.h"
#include "CUDAStream.h"
#include "Ptrchase.h"
#include "CUDAPtrchase.h"


#define CACHELINESIZE 32
int stride = 1;
unsigned int deviceIndex = 0;
bool use_float = false;
int timingsoffset = 1;
unsigned int ARRAY_SIZE = 0;
std::vector<int> pu_ker;
std::vector<int>  puid_ker;
std::array<std::vector<int> ,MAXKERMEMMAPS> pu_mem;
std::array<std::vector<int>,MAXKERMEMMAPS> puid_mem;
int output, kernum;
int ompthreads = 1;
template <typename T>
void check_solution(const unsigned int ntimes, std::vector<T>& a, std::vector<T>& b, std::vector<T>& c);

template <typename T>
void runTPTestUM();

template <typename T>
void runTPTestNonUM();

template <typename T>
void runCudaMemCpyTest();

template <typename T>
void runLatTestNonUM();

template <typename T>
void runLatTestUM();

void parseArguments(int argc, char *argv[]);

void createKerMemVector();

int main(int argc, char *argv[])
{

  ARRAY_SIZE = atoi(argv[1]);
  stride = atoi(argv[2]);
  ompthreads = atoi(argv[3]);
  output = atoi(argv[4]);
  cout << "\n\n......................\n\n";
  cout << "Array size = " << ARRAY_SIZE << " Elements , ( " << ARRAY_SIZE * sizeof(dType)/1024.0/1024.0 << " MB)" << std::endl;
  cout << "Stride = " << stride << std::endl;
  std::cout << "Running kernels " << num_times << " times\n" << std::endl;
  cout << "............................\n " << std::endl;

    if(num_times == 1)
      timingsoffset = 0;

    createKerMemVector();  
    for(int i = 0; i < NOOFSTREAM; i++)
      streamMode.push_back(strMode[i]);
  for (int i = 0; i < pu_ker.size(); i++ )
  {
    kernum = i; 
  if(output == 1)
  {
   if(tpMode == 'Y')
    {
      if(cudaMemMode == 'Y' )
        runCudaMemCpyTest<dType>();
      if(memMode.compare("UM") == 0)
        runTPTestUM<dType>();
      if(memMode.compare("NONUM")== 0)
        runTPTestNonUM<dType>();
      if(memMode.compare("ALL")== 0)
       {
         runTPTestUM<dType>();
 	 runTPTestNonUM<dType>();
  	}
    }
    else {
     cout << "Error: in throughput mode , but throughput flag is off!! Exiting .." << std::endl;
     return 0;
     }
   }
  else
   {
   if(latMode == 'Y')
   {  
     if(memMode.compare("UM")== 0)
        runLatTestUM<dLatType>();
      if(memMode.compare("NONUM")== 0)
        runLatTestNonUM<dLatType>();
      if(memMode.compare("ALL")== 0)
       {
         runLatTestUM<dLatType>();
 	 runLatTestNonUM<dLatType>();
  	}
    }
    else {
      cout << "Error: in latency mode , but latency flag is off!! Exiting .."  << std::endl;
      return 0;
     }
   }
  }

}

void createKerMemVector()
{
 std::string cker,cmem; 
 std::string temp;
 int thw;
 for (int i = 0; i < MAXKERMEMMAPS ; i++)
   {
   // cout << i << std::endl;
    cker = kloc[i];
    if(cker.length() == 0)
     break;
    temp = cker.substr(0,1);
    if(temp.compare("C") == 0)
      thw = 0;
    if(temp.compare("G") == 0)
      thw = 1;
    pu_ker.push_back(thw);
      
    puid_ker.push_back(stoi(cker.substr(1,1)));
    for (int j= 0; j<MAXKERMEMMAPS ; j++)
     {
      cmem = mloc[i][j];
    if(cmem.length() == 0)
     break;
    temp = cmem.substr(0,1);

    if(temp.compare("C") == 0)
      thw = 0;
    if(temp.compare("G") == 0)
      thw = 1;
    pu_mem[i].push_back(thw);
    puid_mem[i].push_back(stoi(cmem.substr(1,1)));
     }
    
   }
}

template <typename T>
void writeFile(std::ofstream *fp, std::ofstream *fpcsv, std::string labels, int size, int stride, int blocksize, int totthread, double  mintime, double maxtime, double average, double actualthroughput, double throughput, int ompthreads)
{


	  std::streamsize ss = std::cout.precision();
         std::string mlocstr, cmem;
	  std::cout.precision(ss);
          for (int j= 0; j<MAXKERMEMMAPS ; j++)
     		{
      			cmem = mloc[kernum][j];
    			if(cmem.length() == 0)
 			{
                         mlocstr=mlocstr.substr(0,mlocstr.length()-1);
    			 break;
                        }
 			mlocstr.append(cmem);	
                        mlocstr.append(",");
		}
	cout << labels << " | " << kloc[kernum] << " | " << mlocstr << " | " << size << " | " << (double) (size * sizeof(T)/1024.0/1024.0) << " | "<< stride << " | " << blocksize << " | " << totthread << " | " << ompthreads << " | " << mintime << " | " << maxtime << " | " << average << " | " << throughput << " \n" << std::endl;
    
	(*fp) << labels <<  " " << kloc[kernum] << " " << mlocstr << " "  << size << " " << (double) (size * sizeof(T)/1024.0/1024.0) << " "<< stride << " " << blocksize << " " << totthread << " " << ompthreads << " " << mintime << " " << maxtime << " " << average << " " << throughput << " " << std::endl;
      (*fpcsv) << labels <<  " " << kloc[kernum] << " " << mlocstr << " "  << size << "," << (double) (size * sizeof(dType)/1024.0/1024.0) << "," << stride << "," << blocksize << "," << totthread << "," << ompthreads << "," << mintime << "," << maxtime << "," << average  <<  "," << throughput << std::endl;

}

template <typename T>
void runCudaMemCpyTest()
{
   std::cout << "Running Test for Cuda mem copy throughput... " << std::endl;
   std::cout <<  std::endl;
  // Create host vectors
    
   std::vector<std::vector<double>> timings(2);
  Stream<T> *stream;
  std::chrono::high_resolution_clock::time_point t1, t2;
   std::vector<string> labels = {"Pinned", "Non Pinned"};

  std::ofstream fp, fpcsv;
  std::ofstream fperror;
  fp.open("./data/result-stream-bw-memcpy.txt", std::ofstream::app);
  fpcsv.open("./data/result-stream-bw-memcpy.csv", std::ofstream::app);
  fpcsv << "DataElement,input size in KB, stride, Block Size, Time in microSec , real Throughput in GByte/Sec, Throughput (GByte/sec)" << std::endl;

  // Use the CUDA implementation
  stream = new CUDAStream<T>(ARRAY_SIZE, stride,  pu_ker.at(kernum), puid_ker.at(kernum),pu_mem[kernum],puid_mem[kernum]);
  stream->allocate_arrays_cudamem();
 //stream->enablePeerAccess();
  
  for (unsigned int k = 0; k < num_times; k++)
  {
       
     t1 = std::chrono::high_resolution_clock::now();
    stream->cudaMemCopyPinned();
     t2 = std::chrono::high_resolution_clock::now();
    timings[0].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

     t1 = std::chrono::high_resolution_clock::now();
    stream->cudaMemCopyNonPinned();
     t2 = std::chrono::high_resolution_clock::now();
    timings[1].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
   
  }


  for (int i = 0; i < 2; i++)
  {
    // Get min/max; ignore the first result
    auto minmax = std::minmax_element(timings[i].begin()+timingsoffset, timings[i].end());

    // Calculate average; ignore the first result
    double average = std::accumulate(timings[i].begin()+timingsoffset, timings[i].end(), 0.0) / (double)(num_times);

    double mintime =  (*minmax.first);
    if (mintime <= 0.0)
       mintime = *minmax.first;
     double throughput = ARRAY_SIZE * sizeof(T) * 1e-9 / mintime;

     double actualthroughput = throughput;

     writeFile<T>(&fp, &fpcsv, labels[i], ARRAY_SIZE, stride, 1, 1, *minmax.first, *minmax.second ,  average, actualthroughput, throughput,0 );      

   }
  
  //stream->freeCudaMem();
  delete stream;
  fp.close();
  fpcsv.close();
  //CudaContext.ProfilerStop();
}



template <typename T>
void runLatTestNonUM()
{
   std::cout << "\nRunning Latency Test without Unified Memory... " << std::endl;
   std::cout <<  std::endl;
  
  if((pu_ker.at(kernum) == 0 && pu_mem[kernum].at(0) != 0 )|| pu_mem[kernum].size() != 1)
    {
     cout << "Error: Invalid kernel memory mapping for Non UM case!! Returning to the next test!!" << std::endl;
     return;
    }
 
  Ptrchase<T> *pchase;
  std::ofstream fp, fpcsv;
  std::ofstream fperror;
  fp.open("./data/result-pchase-lat-nonum.txt", std::ofstream::app);
  fpcsv.open("./data/result-pchase-lat-nonum.csv", std::ofstream::app);
  //fpcsv << "DataElement,input size in KB, stride, Block Size, Time in microSec , real Throughput in GByte/Sec, Throughput (GByte/sec)" << std::endl;

  // Use the CUDA implementation
  pchase = new CUDAPtrchase<T>(ARRAY_SIZE, stride, iterations,  pu_ker.at(kernum), puid_ker.at(kernum),pu_mem[kernum],puid_mem[kernum]);
  pchase->enablePeerAccess();
  // List of times
  std::vector<double> timings;
  double time;
  // Declare timers
  
  //std::string labels[NOOFSTREAM];
   // Main loop
  //
//   cout << "Latency =  | KERNEL | MEMORY | #-ELEMENTS | DATASIZE(MB) | Stride | BlockSize | GPUThreads | MinTime(μs) | MaxTime(μs) | AvgTime(μs) | Latency(μs)" << std::endl;
  // cout << std::endl;
  // fp << "Latemcy KERNEL MEMORY #-ELEMENTS DATA-SIZE(MB) Stride Block-Size GPU-Threads Min-Time(μs) MaxTime(μs) AvgTime(μs) Latency(μs)" << std::endl;
   cout << "Mode =  | KERNEL | MEMORY | #-ELEMENTS | DATASIZE(MB) | Stride | BlockSize | GPUThreads | OMPTHREADS | AvgCycle | MinLatency(μs) | MaxLatency(μs) | AvgLatency(μs)" << std::endl;
   cout << std::endl;
   //fp << "Mode KERNEL MEMORY #-ELEMENTS DATA-SIZE(MB) Stride Block-Size GPU-Threads OMPTHREADS AvgCycle MaxLatency(μs) MinLatency(μs) AvgLatency(μs)" << std::endl;

   
  for (unsigned int k = 0; k < num_times; k++)
  {
  pchase->allocate_arrays_nonum();
  pchase->init_arrays_num();
    time = pchase->latency();
    timings.push_back(time);  
    pchase->freeMemNonUM();
  
  }


  for (int i = 0; i < 1; i++)
  {
    // Get min/max; ignore the first result
    auto minmax = std::minmax_element(timings.begin(), timings.end());

    // Calculate average; ignore the first result
    double average = std::accumulate(timings.begin(), timings.end(), 0.0) / (double)(num_times);

    

    double mintime =  (*minmax.first);
    if (mintime <= 0.0)
       mintime = *minmax.first;
	
    double avgcycle = average / ARRAY_SIZE/iterations;
    double mincycle = (double) *minmax.first/ ARRAY_SIZE/iterations;
    double maxcycle = (double) *minmax.second/ ARRAY_SIZE/iterations;
    double minlatency = mincycle * 1e+6 ;
    double maxlatency = maxcycle * 1e+6 ;
    double avglatency = avgcycle * 1e+6 ;

    writeFile<T>(&fp, &fpcsv, "Latency = ", ARRAY_SIZE, stride, 1,1, avgcycle , minlatency,  maxlatency, 0 , avglatency,0 );
    //   writeFile<T>(&fp, &fpcsv, "Latency = ", ARRAY_SIZE, stride, 1,1, mincycle , maxcycle,  minlatency, maxlatency , avglatency,0 );   
  }
  //pchase->freeMemNonUM();
  pchase->disablePeerAccess();
  delete pchase;
  fp.close();
  fpcsv.close();
  //CudaContext.ProfilerStop();
}

template <typename T>
void runLatTestUM()
{
   std::cout << "\nRunning Latency Test with Unified Memory... " << std::endl;
   std::cout <<  std::endl;
  Ptrchase<T> *pchase;
  std::ofstream fp, fpcsv;
  std::ofstream fperror;
  fp.open("./data/result-pchase-lat-um.txt", std::ofstream::app);
  fpcsv.open("./data/result-pchase-lat-um.csv", std::ofstream::app);
  
   cout << "Mode =  | KERNEL | MEMORY | #-ELEMENTS | DATASIZE(MB) | Stride | BlockSize | GPUThreads | OMPTHREADS | AvgCycle | MinLatency(μs) | MaxLatency(μs) | AvgLatency(μs)" << std::endl;
   cout << std::endl;
   //fp << "Mode= KERNEL MEMORY #-ELEMENTS DATA-SIZE(MB) Stride Block-Size GPU-Threads OMPTHREADS AvgCycle MaxLatency(μs) MinLatency(μs) AvgLatency(μs)" << std::endl;

  // Use the CUDA implementation
  pchase = new CUDAPtrchase<T>(ARRAY_SIZE, stride, iterations, pu_ker.at(kernum), puid_ker.at(kernum),pu_mem[kernum],puid_mem[kernum]);
  
   pchase->enablePeerAccess();

  // List of times
  std::vector<float> timings;
  float time;
  // Declare timers
  
  for (unsigned int k = 0; k < num_times; k++)
  {
    pchase->allocate_arrays_um();
    pchase->init_arrays_um();
    pchase->distributeUMemory();
    time = pchase->latency();
    timings.push_back(time); 
    pchase->freeMemUM(); 
  }


  for (int i = 0; i < 1; i++)
  {
    // Get min/max; ignore the first result
    auto minmax = std::minmax_element(timings.begin(), timings.end());

    // Calculate average; ignore the first result
    double average = std::accumulate(timings.begin(), timings.end(), 0.0) / (double)(num_times);

    double mintime =  (*minmax.first);
    if (mintime <= 0.0)
       mintime = *minmax.first;
	
    double avgcycle = average / ARRAY_SIZE/iterations;
    double mincycle = (double) *minmax.first/ ARRAY_SIZE/iterations;
    double maxcycle = (double) *minmax.second/ ARRAY_SIZE/iterations;
    double minlatency = mincycle * 1e+6 ;
    double maxlatency = maxcycle * 1e+6;
    double avglatency = avgcycle * 1e+6;
    writeFile<T>(&fp, &fpcsv, "Latency = ", ARRAY_SIZE, stride, 1,1, avgcycle , minlatency,  maxlatency, 0 , avglatency,0 );

//    writeFile<T>(&fp, &fpcsv, "Latency = ", ARRAY_SIZE, stride, 1, 1, avgcycle,  minlatency, maxlatency , avglatency,0 ,0);   
  }
 //  pchase->freeMemUM();
   pchase->disablePeerAccess();
   delete pchase;
  fp.close();
  fpcsv.close();
  //CudaContext.ProfilerStop();
}


template <typename T>
void runTPTestNonUM()
{
   std::cout << "\nRunning Throughput Test without Unified Memory... " << std::endl;
   std::cout <<  std::endl;
   if(pu_ker.at(kernum) == 0 || pu_mem[kernum].size() != 1)
    {
     cout << "Error: Invalid kernel memory mapping for Non UM case!! Returning to the next test!!" << std::endl;
     return;
    }
 /*
  if (sizeof(T) == sizeof(float))
    std::cout << "Precision: float" << std::endl;
  else
    std::cout << "Precision: double" << std::endl;
 */
  // Create host vectors
  Stream<T> *stream;
  std::chrono::high_resolution_clock::time_point t1, t2;


  std::ofstream fp, fpcsv;
  std::ofstream fperror;
  fp.open("./data/result-stream-bw-nonum.txt", std::ofstream::app);
  fpcsv.open("./data/result-stream-bw-nonmum.csv", std::ofstream::app);
  fpcsv << "DataElement,input size in KB, stride, Block Size, Time in microSec , real Throughput in GByte/Sec, Throughput (GByte/sec)" << std::endl;
 //   cout << "STREAM KERNEL | MEMORY | # of ELEMENTS | DATA SIZE (MB) | Stride | Block Size | GPU Threads | Min Time(μs) | Max Time(μs) | Avg Time(μs) | Throughput GB/s" << std::endl;
   cout << "STREAM | KERNEL | MEMORY | #-ELEMENTS | DATASIZE(MB) | Stride | BlockSize | GPUThreads | MinTime(sec) | MaxTime(sec) | AvgTime(sec) | Throughput(GB/s)" << std::endl;
  cout << std::endl;
   //fp << "STREAM KERNEL MEMORY #-ELEMENTS | DATA-SIZE(MB) Stride Block-Size GPU-Threads Min-Time(sec) MaxTime(sec) AvgTime(sec) Throughput(GB/s)" << std::endl;
  // Use the CUDA implementation
  stream = new CUDAStream<T>(ARRAY_SIZE, stride, pu_ker.at(kernum), puid_ker.at(kernum),pu_mem[kernum],puid_mem[kernum]);
  stream->allocate_arrays_nonum();
  stream->init_arrays_num();
  
  stream->distributeNUMemory(); 
  stream->setOMPParams(ompthreads);
  stream->enablePeerAccess();
  // List of times
  std::vector<std::vector<double>> timings(NOOFSTREAM);

  // Declare timers
   double time;
  
  //std::string labels[NOOFSTREAM];
  size_t sizes[NOOFSTREAM];
  
  int j = 0;
  if(std::find(streamMode.begin(), streamMode.end(), "READ") != streamMode.end())
   sizes[j++] = sizeof(T) * ARRAY_SIZE;
 
   if(std::find(streamMode.begin(), streamMode.end(), "WRITE") != streamMode.end())
   sizes[j++] = sizeof(T) * ARRAY_SIZE;
 
   if(std::find(streamMode.begin(), streamMode.end(), "COPY") != streamMode.end())
   sizes[j++] = 2 * sizeof(T) * ARRAY_SIZE;
 
   if(std::find(streamMode.begin(), streamMode.end(), "SCALE") != streamMode.end())
   sizes[j++] =  2 * sizeof(T) * ARRAY_SIZE;

     if(std::find(streamMode.begin(), streamMode.end(), "ADD") != streamMode.end())
   sizes[j++] = 3 * sizeof(T) * ARRAY_SIZE;

   if(std::find(streamMode.begin(), streamMode.end(), "TRIAD") != streamMode.end())
   sizes[j++] = 3 * sizeof(T) * ARRAY_SIZE;
   
      if(std::find(streamMode.begin(), streamMode.end(), "QUADAD") != streamMode.end())
	   sizes[j++] = 4 * sizeof(T) * ARRAY_SIZE;

	 if(std::find(streamMode.begin(), streamMode.end(), "PENTAD") != streamMode.end())
           sizes[j++] = 5 * sizeof(T) * ARRAY_SIZE;

         if(std::find(streamMode.begin(), streamMode.end(), "HEXAD") != streamMode.end())
           sizes[j++] = 6 * sizeof(T) * ARRAY_SIZE;

   int startThread;
  // Main loop
  //
  if (exp_thread == 'Y')
     {
       startThread = 1;
      // endThread = ARRAY_SIZE;
      }
   else
    {
      startThread = ARRAY_SIZE;
    }
   
  for(int blockSize= startBlock ; blockSize<=endBlock ; blockSize*=2 )
  {
  for(int thread = startThread; thread <=ARRAY_SIZE; thread = thread * 2)
      {
   stream->setDeviceParameterNUM(blockSize, thread);
	
//   stream->setDeviceParameterNUM(blockSize);
  for (unsigned int k = 0; k < num_times; k++)
  {
    j = 0;
   if(pu_ker.at(kernum) == 1)
   {
   if(std::find(streamMode.begin(), streamMode.end(), "READ") != streamMode.end())
   {
    t1 = std::chrono::high_resolution_clock::now();
    stream->num_read();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
   }  
   
    if(std::find(streamMode.begin(), streamMode.end(), "WRITE") != streamMode.end())
    { 
    t1 = std::chrono::high_resolution_clock::now();
    stream->num_write();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count()); 
    }
    
    if(std::find(streamMode.begin(), streamMode.end(), "COPY") != streamMode.end()) {
    t1 = std::chrono::high_resolution_clock::now();
     stream->num_copy();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
   }
    
    if(std::find(streamMode.begin(), streamMode.end(), "SCALE") != streamMode.end()){
        t1 = std::chrono::high_resolution_clock::now();
    stream->num_mul();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    }
   
    if(std::find(streamMode.begin(), streamMode.end(), "ADD") != streamMode.end()) 
     {
     t1 = std::chrono::high_resolution_clock::now();
    stream->num_add();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
   }
   
   if(std::find(streamMode.begin(), streamMode.end(), "TRIAD") != streamMode.end()) {
        t1 = std::chrono::high_resolution_clock::now();
    stream->num_triad();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    }
   
   if(std::find(streamMode.begin(), streamMode.end(), "QUADAD") != streamMode.end()) {
        t1 = std::chrono::high_resolution_clock::now();
    stream->num_quadad();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    }
    
    if(std::find(streamMode.begin(), streamMode.end(), "PENTAD") != streamMode.end()) {
        t1 = std::chrono::high_resolution_clock::now();
    stream->num_pentad();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    }
   
    if(std::find(streamMode.begin(), streamMode.end(), "HEXAD") != streamMode.end()) {
        t1 = std::chrono::high_resolution_clock::now();
    stream->num_hexad();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    }
   }
  
  }

  
  for (int i = 0; i < NOOFSTREAM; i++)
  {
    // Get min/max; ignore the first result
    auto minmax = std::minmax_element(timings[i].begin()+timingsoffset, timings[i].end());

    // Calculate average; ignore the first result
    double average = std::accumulate(timings[i].begin()+timingsoffset, timings[i].end(), 0.0) / (double) (num_times - timingsoffset);
    timings[i].clear();
    double mintime =  (*minmax.first);
    if (mintime <= 0.0)
       mintime = *minmax.first;
     double throughput = sizes[i] * 1e-9 / mintime;

     double actualthroughput = throughput;

     if(stride * sizeof(dType) >= CACHELINESIZE )
          {
           // actualthroughput = ARRAY_SIZE * CACHELINESIZE * 1e-9 / *minmax.first;
          }

       writeFile<T>(&fp, &fpcsv, streamMode.at(i), ARRAY_SIZE, stride, blockSize, thread ,*minmax.first , *minmax.second,  average, actualthroughput, throughput,0 );

   }
   }
  }
  //cout << "test " << std::endl;
  stream->freeMemNonUM();
  stream->disablePeerAccess();
  delete stream;
  fp.close();
  fpcsv.close();
   
  //CudaContext.ProfilerStop();
}



template <typename T>
void runTPTestUM()
{
   std::cout << "\nRunning Throughput Test with Unified Memory... " << std::endl;
   std::cout <<  std::endl;
 /*
  if (sizeof(T) == sizeof(float))
    std::cout << "Precision: float" << std::endl;
  else
    std::cout << "Precision: double" << std::endl;
 */

  Stream<T> *stream;
  std::chrono::high_resolution_clock::time_point t1, t2;


  std::ofstream fp, fpcsv;
  std::ofstream fperror;
  fp.open("./data/result-stream-bw-um.txt", std::ofstream::app);
  fpcsv.open("./data/result-stream-bw-um.csv", std::ofstream::app);
  fpcsv << "DataElement,input size in KB, stride, Block Size, Time in microSec , real Throughput in GByte/Sec, Throughput (GByte/sec)" << std::endl;
    // cout << "STREAM KERNEL MEMORY #-ELEMENTS DATASIZE(MB) Stride Block Size GPUThreads MinTime(μs) MaxTime(μs) Avg Time(μs) Throughput GB/s" << std::endl;
     cout << "STREAM | KERNEL | MEMORY | #-ELEMENTS | DATASIZE(MB) | Stride | BlockSize | GPUThreads | MinTime(sec) | MaxTime(sec) | AvgTime(sec) | Throughput(GB/s)" << std::endl;
     cout << std::endl; 
  //fp << "STREAM KERNEL MEMORY #-ELEMENTS DATA-SIZE(MB) Stride Block-Size GPU-Threads Min-Time(μs) MaxTime(μs) AvgTime(μs) Throughput(GB/s)" << std::endl;

  // Use the CUDA implementation
  stream = new CUDAStream<T>(ARRAY_SIZE, stride, pu_ker.at(kernum), puid_ker.at(kernum),pu_mem[kernum],puid_mem[kernum]);
  stream->allocate_arrays_um(memAdvise);
  stream->init_arrays_um();
  stream->setOMPParams(ompthreads);  
 // stream->enablePeerAccess(); 
 // stream->setMemadvise(memAdvise);
  // List of times
  std::vector<std::vector<double>> timings(NOOFSTREAM);

  // Declare timers
   double time;
  
  //std::string labels[NOOFSTREAM];
  size_t sizes[NOOFSTREAM];
  
  int j = 0;
  if(std::find(streamMode.begin(), streamMode.end(), "READ") != streamMode.end())
   sizes[j++] = sizeof(T) * ARRAY_SIZE;
 
   if(std::find(streamMode.begin(), streamMode.end(), "WRITE") != streamMode.end())
   sizes[j++] = sizeof(T) * ARRAY_SIZE;
 
   if(std::find(streamMode.begin(), streamMode.end(), "COPY") != streamMode.end())
   sizes[j++] = 2 * sizeof(T) * ARRAY_SIZE;
 
   if(std::find(streamMode.begin(), streamMode.end(), "SCALE") != streamMode.end())
   sizes[j++] =  2 * sizeof(T) * ARRAY_SIZE;

     if(std::find(streamMode.begin(), streamMode.end(), "ADD") != streamMode.end())
   sizes[j++] = 3 * sizeof(T) * ARRAY_SIZE;

   if(std::find(streamMode.begin(), streamMode.end(), "TRIAD") != streamMode.end())
   sizes[j++] = 3 * sizeof(T) * ARRAY_SIZE;
   
    if(std::find(streamMode.begin(), streamMode.end(), "QUADAD") != streamMode.end())
	   sizes[j++] = 4 * sizeof(T) * ARRAY_SIZE;

    if(std::find(streamMode.begin(), streamMode.end(), "PENTAD") != streamMode.end())
           sizes[j++] = 5 * sizeof(T) * ARRAY_SIZE;

   if(std::find(streamMode.begin(), streamMode.end(), "HEXAD") != streamMode.end())
           sizes[j++] = 6 * sizeof(T) * ARRAY_SIZE;
   
   /*  if(std::find(streamMode.begin(), streamMode.end(), "HCOPY") != streamMode.end())
           sizes[j++] = 2 * sizeof(T) * ARRAY_SIZE;

   if(std::find(streamMode.begin(), streamMode.end(), "HADD") != streamMode.end())
           sizes[j++] = 3 * sizeof(T) * ARRAY_SIZE;*/

  // Main loop
  //
// T * temp = (T *) malloc(sizeof(T));
 int startThread;	
 if (exp_thread == 'Y')
     {
       startThread = 1;
      // endThread = ARRAY_SIZE;
     }
     else
     {
      startThread = ARRAY_SIZE;
     }
     
 for(int blockSize= startBlock ; blockSize<=endBlock ; blockSize*=2 )
  {
  for(long long thread = startThread; thread <=ARRAY_SIZE; thread = thread * 2)
    {
      stream->setDeviceParameterUM(blockSize, thread);
  //    stream->setDeviceParameterUM(blockSize);
  // stream->distributeUMemory();
  for (unsigned int k = 0; k < num_times; k++)
  {
    j = 0;
   if(pu_ker.at(kernum) == 1)
   {
   if(std::find(streamMode.begin(), streamMode.end(), "READ") != streamMode.end())
   {
    stream->distributeUMemory();     
    t1 = std::chrono::high_resolution_clock::now();
    stream->read();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
   }  
   
    if(std::find(streamMode.begin(), streamMode.end(), "WRITE") != streamMode.end())
    { 

    stream->distributeUMemory();
    t1 = std::chrono::high_resolution_clock::now();
    stream->write();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count()); 
    }
    
    if(std::find(streamMode.begin(), streamMode.end(), "COPY") != streamMode.end()) {
    //cout << " copy again " << k << std::endl;
    stream->distributeUMemory();
    t1 = std::chrono::high_resolution_clock::now();
     stream->copy();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
   }
    
    if(std::find(streamMode.begin(), streamMode.end(), "SCALE") != streamMode.end()) 
    {
     stream->distributeUMemory();
    t1 = std::chrono::high_resolution_clock::now();
    stream->mul();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    }
   
    if(std::find(streamMode.begin(), streamMode.end(), "ADD") != streamMode.end()) 
     {
    stream->distributeUMemory();
    t1 = std::chrono::high_resolution_clock::now();
    stream->add();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
   }
   
   if(std::find(streamMode.begin(), streamMode.end(), "TRIAD") != streamMode.end())
    {
    stream->distributeUMemory();
    t1 = std::chrono::high_resolution_clock::now();
    stream->triad();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    }

   if(std::find(streamMode.begin(), streamMode.end(), "QUADAD") != streamMode.end())
    {
    stream->distributeUMemory();
    t1 = std::chrono::high_resolution_clock::now();
    stream->quadad();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    }

    if(std::find(streamMode.begin(), streamMode.end(), "PENTAD") != streamMode.end())
    {
    stream->distributeUMemory();
    t1 = std::chrono::high_resolution_clock::now();
    stream->pentad();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    }
    
    if(std::find(streamMode.begin(), streamMode.end(), "HEXAD") != streamMode.end())
    {
    stream->distributeUMemory();
    t1 = std::chrono::high_resolution_clock::now();
    stream->hexad();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    }
   }
   if(pu_ker.at(kernum)== 0)
   {
   if(std::find(streamMode.begin(), streamMode.end(), "READ") != streamMode.end())
    {
  //  temp = (T *) malloc(sizeof(T));
    stream->distributeUMemory();
    t1 = std::chrono::high_resolution_clock::now();
    stream->h_read();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    }

    if(std::find(streamMode.begin(), streamMode.end(), "WRITE") != streamMode.end())
    {
    stream->distributeUMemory();
    t1 = std::chrono::high_resolution_clock::now();
    stream->h_write();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    }
      if(std::find(streamMode.begin(), streamMode.end(), "COPY") != streamMode.end())
    {
    stream->distributeUMemory();
    t1 = std::chrono::high_resolution_clock::now();
    stream->h_copy();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    }

    if(std::find(streamMode.begin(), streamMode.end(), "SCALE") != streamMode.end())
    {
    stream->distributeUMemory();
    t1 = std::chrono::high_resolution_clock::now();
    stream->h_mul();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    }
      if(std::find(streamMode.begin(), streamMode.end(), "ADD") != streamMode.end())
    {
    stream->distributeUMemory();
    t1 = std::chrono::high_resolution_clock::now();
    stream->h_add();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    }

    if(std::find(streamMode.begin(), streamMode.end(), "TRIAD") != streamMode.end())
    {
    stream->distributeUMemory();
    t1 = std::chrono::high_resolution_clock::now();
    stream->h_triad();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    }
   if(std::find(streamMode.begin(), streamMode.end(), "QUADAD") != streamMode.end())
    {
    stream->distributeUMemory();
    t1 = std::chrono::high_resolution_clock::now();
    stream->h_quadad();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    }

    if(std::find(streamMode.begin(), streamMode.end(), "PENTAD") != streamMode.end())
    {
    stream->distributeUMemory();
    t1 = std::chrono::high_resolution_clock::now();
    stream->h_pentad();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    }
  
      if(std::find(streamMode.begin(), streamMode.end(), "HEXAD") != streamMode.end())
    {
    stream->distributeUMemory();
    t1 = std::chrono::high_resolution_clock::now();
    stream->h_hexad();
    t2 = std::chrono::high_resolution_clock::now();
    timings[j++].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    }
   }   
  }
  
  for (int i = 0; i < NOOFSTREAM; i++)
  {
    // Get min/max; ignore the first result
    auto minmax = std::minmax_element(timings[i].begin()+timingsoffset, timings[i].end());

    // Calculate average; ignore the first result
    double average = std::accumulate(timings[i].begin()+timingsoffset, timings[i].end(), 0.0) / (double)(num_times - timingsoffset);
    timings[i].clear();
    double mintime =  (*minmax.first);
    if (mintime <= 0.0)
       mintime = *minmax.first;
     double throughput = sizes[i] * 1e-9 / mintime;

     double actualthroughput = throughput;
     
       writeFile<T>(&fp, &fpcsv, streamMode.at(i), ARRAY_SIZE, stride, blockSize, thread, *minmax.first, *minmax.second, average, actualthroughput, throughput, ompthreads );
       

   }
   }
  }
  //stream->disablePeerAccess();
  stream->freeMemUM();
  delete stream;
  fp.close();
  fpcsv.close();
  //CudaContext.ProfilerStop();
}



template <typename T>
void check_solution(const unsigned int ntimes, std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  // Generate correct solution
  T goldA = startA;
  T goldB = startB;
  T goldC = startC;

  const T scalar = startScalar;

  for (unsigned int i = 0; i < ntimes; i++)
  {
    // Do STREAM!
    goldC = goldA;
    goldB = scalar * goldC;
    goldC = goldA + goldB;
    goldA = goldB + scalar * goldC;
  }
 int startThread;
  // Calculate the average error
  double errA = std::accumulate(a.begin(), a.end(), 0.0, [&](double sum, const T val){ return sum + fabs(val - goldA); });
  errA /= a.size();
  double errB = std::accumulate(b.begin(), b.end(), 0.0, [&](double sum, const T val){ return sum + fabs(val - goldB); });
  errB /= b.size();
  double errC = std::accumulate(c.begin(), c.end(), 0.0, [&](double sum, const T val){ return sum + fabs(val - goldC); });
  errC /= c.size();

  double epsi = std::numeric_limits<T>::epsilon() * 100.0;

  if (errA > epsi)
    std::cerr
      << "Validation failed on a[]. Average error " << errA
      << std::endl;
  if (errB > epsi)
    std::cerr
      << "Validation failed on b[]. Average error " << errB
      << std::endl;
  if (errC > epsi)
    std::cerr
      << "Validation failed on c[]. Average error " << errC
      << std::endl;

}

int parseUInt(const char *str, unsigned int *output)
{
  char *next;
  *output = strtoul(str, &next, 10);
  return !strlen(next);
}

void parseArguments(int argc, char *argv[])
{
  for (int i = 1; i < argc; i++)
  {
    if (!std::string("--list").compare(argv[i]))
    {
      listDevices();
      exit(EXIT_SUCCESS);
    }
    else if (!std::string("--device").compare(argv[i]))
    {
      if (++i >= argc || !parseUInt(argv[i], &deviceIndex))
      {
        std::cerr << "Invalid device index." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--arraysize").compare(argv[i]) ||
             !std::string("-s").compare(argv[i]))
    {
      if (++i >= argc || !parseUInt(argv[i], &ARRAY_SIZE))
      {
        std::cerr << "Invalid array size." << std::endl;
        exit(EXIT_FAILURE);
        std::cerr << "Number of times must be 2 or more" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--float").compare(argv[i]))
    {
      use_float = true;
    }
    else if (!std::string("--help").compare(argv[i]) ||
             !std::string("-h").compare(argv[i]))
    {
      std::cout << std::endl;
      std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "  -h  --help               Print the message" << std::endl;
      std::cout << "      --list               List available devices" << std::endl;
      std::cout << "      --device     INDEX   Select device at INDEX" << std::endl;
      std::cout << "  -s  --arraysize  SIZE    Use SIZE elements in the array" << std::endl;
      std::cout << "  -n  --numtimes   NUM     Run the test NUM times (NUM >= 2)" << std::endl;
      std::cout << "      --float              Use floats (rather than doubles)" << std::endl;
      std::cout << std::endl;
      exit(EXIT_SUCCESS);
    }
    else
    {
      std::cerr << "Unrecognized argument '" << argv[i] << "' (try '--help')"
                << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}
