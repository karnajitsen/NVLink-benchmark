
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
#include "Ptrchase.h"

#define IMPLEMENTATION_STRING "CUDA"

template <class T>
class CUDAPtrchase : public Ptrchase<T>
{
  protected:
    // Size of arrays
    unsigned int chase_elem;
   int stride;
     int blockSize;
     int gridSize;	    // Device side pointers to arrays
     int puker, puidker;
    vector<int> pumem, puidmem,indexchunk;
    T *d_a;
    T *d_b;
    T *h_a, *h_b;
    T *xj, *xi;
    double *d_time, h_time;	
    unsigned int iterations;
  public:

    CUDAPtrchase(const unsigned int, const int, const int,int, int ,vector<int>, vector<int>);
    ~CUDAPtrchase();
    virtual double latency() override;
    virtual void enablePeerAccess() override;
    virtual void disablePeerAccess() override; 
    virtual void init_arrays_um() override;
    virtual void init_arrays_num() override;
    virtual void allocate_arrays_um() override;
    virtual void allocate_arrays_nonum() override;
    virtual void allocate_arrays_device_lat() override;
    virtual void freeMemNonUM() override;
    virtual void freeMemUM() override;
    virtual void freeCudaMem() override;
    virtual void write_arrays() override;
    virtual void distributeUMemory() override;
};
