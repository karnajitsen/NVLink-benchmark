
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
#define startScalar (0.4)

template <class T>
class Ptrchase
{
  public:

    virtual ~Ptrchase(){}

    // Kernels
    // These must be blocking calls
        virtual double latency() = 0;

    // Copy memory between host and device
    virtual void init_arrays_um() = 0;
      virtual void init_arrays_num() = 0;
    virtual void allocate_arrays_um() = 0;
    virtual void allocate_arrays_nonum() = 0;
    virtual void allocate_arrays_device_lat() = 0;
        virtual void write_arrays() = 0;
     virtual void freeMemNonUM() = 0;
    virtual void freeMemUM() = 0;
    virtual void freeCudaMem() = 0;
    virtual void distributeUMemory() = 0;
    virtual void enablePeerAccess() = 0;
    virtual void disablePeerAccess() = 0;

};


// Implementation specific device functions
void listDevices(void);
std::string getDeviceName(const int);
std::string getDeviceDriver(const int);

