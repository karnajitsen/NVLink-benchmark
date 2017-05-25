
// Copyright (c) 2016-17 Karnajit Sen
// // IBM Deutschland R&D GmbH, University of Erlangen-Nürnberg
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <vector>
#include <string>

// Array values
#define startA (0.1)
#define startB (0.2)
#define startC (0.3)
#define startD (0.4)
#define startE (0.5)
#define startF (0.6)
#define startScalar (0.4)

template <class T>
class Stream
{
  public:

    virtual ~Stream(){}

    // Kernels
    // These must be blocking calls
    virtual void read() = 0;
    virtual void write() = 0;
    virtual void copy() = 0;
    virtual void mul() = 0;
    virtual void add() = 0;
    virtual void triad() = 0;
    virtual void quadad() = 0;
    virtual void pentad() = 0;
    virtual void hexad() = 0;
     virtual void num_read() = 0;
    virtual void num_write() = 0;
    virtual void num_copy() = 0;
    virtual void num_mul() = 0;
    virtual void num_add() = 0;
    virtual void num_triad() = 0;
     virtual void num_quadad() = 0;
    virtual void num_pentad() = 0;
    virtual void num_hexad() = 0;
    virtual void cudaMemCopyPinned() = 0;
    virtual void cudaMemCopyNonPinned() = 0;
    virtual void h_read() = 0;
    virtual void h_write() = 0;
    virtual void h_copy() = 0;
    virtual void h_mul() = 0;
    virtual void h_add() = 0;
    virtual void h_triad() = 0;
    virtual void h_quadad() = 0;
    virtual void h_pentad() = 0;
    virtual void h_hexad() = 0;

    
    // Copy memory between host and device
    virtual void init_arrays_um() = 0;
    virtual void init_arrays_num() = 0;

    virtual void allocate_arrays_um(char) = 0;
    virtual void allocate_arrays_nonum() = 0;
    virtual void allocate_arrays_membw() = 0;
    virtual void allocate_arrays_cudamem() = 0;
    virtual void read_arrays_num() = 0;
    virtual void setDeviceParameterUM(int bs,int thread) = 0 ;
    virtual void setDeviceParameterNUM(int bs,int thread) = 0 ;
    virtual void setApplictionClock() = 0;
    virtual void lockMem() = 0;
    virtual void freeMemNonUM() = 0;
    virtual void freeMemUM() = 0;
    virtual void freeCudaMem() = 0;
    virtual void setOMPParams(int cputhreads) = 0;
    virtual void distributeUMemory() = 0;
    virtual void distributeNUMemory() = 0; 
    virtual void enablePeerAccess() = 0;
    virtual void disablePeerAccess() = 0;
};


// Implementation specific device functions
void listDevices(void);
std::string getDeviceName(const int);
std::string getDeviceDriver(const int);

