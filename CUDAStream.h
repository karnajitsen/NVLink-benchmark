
// Copyright (c) 2016-17 Karnajit Sen
// University of Erlangen-Nürnberg
// 
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>
#include <sstream>
using namespace std;
#include "Stream.h"

#define IMPLEMENTATION_STRING "CUDA"

template <class T>
class CUDAStream : public Stream<T>
{
  protected:
    // Size of arrays
    unsigned int array_size;
    int stride;
    int blockSize;
    std::vector<int> gridSize;	    // Device side pointers to arrays
    int threadSize;
     int puker, puidker;
    vector<int> pumem, puidmem,indexchunk;
     T *d_a, *d_b, *d_c, *d_d, * d_e, * d_f;
     T *h_a, *h_b, *h_c, *h_d, * h_e, *h_f;
     T **num_a, **num_b, **num_c, **num_d, **num_e, **num_f;	
    T **dnum_a, **dnum_b, **dnum_c, **dnum_d, **dnum_e, **dnum_f;
   // __global__ int j;
  public:

    CUDAStream(const unsigned int, int, int , int ,vector<int>, vector<int>);
    ~CUDAStream();
    virtual void read() override;
    virtual void write() override;
    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual void quadad() override;
    virtual void pentad() override;
    virtual void hexad() override;
     virtual void num_read() override;
    virtual void num_write() override;
    virtual void num_copy() override;
    virtual void num_add() override;
    virtual void num_mul() override;
    virtual void num_triad() override;
    virtual void num_quadad() override;
    virtual void num_pentad() override;
    virtual void num_hexad() override;
    virtual void cudaMemCopyPinned() override;
    virtual void cudaMemCopyNonPinned() override;
    virtual void h_read() override;
    virtual void h_write() override;
    virtual void h_copy() override;
    virtual void h_add() override;
    virtual void h_mul() override;
    virtual void h_triad() override;
    virtual void h_quadad() override;
    virtual void h_pentad() override;
    virtual void h_hexad() override;
    virtual void lockMem() override;
    virtual void setApplictionClock() override; 
    virtual void init_arrays_um() override;
    virtual void init_arrays_num() override;
    virtual void allocate_arrays_um(const char memAdvise) override;
    virtual void allocate_arrays_nonum() override;
    virtual void allocate_arrays_membw() override;
    virtual void allocate_arrays_cudamem() override;
    virtual void freeMemNonUM() override;
    virtual void freeMemUM() override;
    virtual void freeCudaMem() override;
    virtual void read_arrays_num() override;
    virtual void setDeviceParameterUM(int bs,int thread) override;
    virtual void setDeviceParameterNUM(int bs,int thread) override;
    virtual void setOMPParams(int cputhreads) override;
    virtual void distributeUMemory() override;
    virtual void distributeNUMemory() override;
    virtual void enablePeerAccess() override;
    virtual void disablePeerAccess() override;
};
