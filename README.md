Configuration:

/config/parameters.config file contains all the parameters of measurement to be configured. In our parameter file, we have used following letters to denote unit of data size,
B = Byte
K = Kilo Bytes
M = Mega Bytes
G = Giga Bytes

An example of parameters and there description are as follows,

TPDATATYPE = double                                    : data type for througput experiement
LATDATATYPE = int                                      : data type for latency experiement
SIZE = 32M-8G,2                                        : Size of the array, data size. format - 32 = Starting minimum data size, 8 = Last maximum data size, "-" = Minimum and maximum data size separator, "," = data size and incrementing factor separator, "2" = incrementing factor multiple of 2,
ACCESSPATTERN = LINEAR                                 : Access pattern, not used so far
THROUGHPUT = N                                         : "Y" = throughput experiment will takes place, if "N" no throughput experiements
CUDAMEMCOPY = N                                        : "Y" = cudamemcpy experiment will takes place, if "N" no cudamemcpy experiements
KER2MEMLOC = C0-C0;C0-G0;G0-G1; C0-G0-G1               : "Kernel location"-Memory location 1, Memory Location 2, ... ", Mapping Separator is ";"
STREAM = READ,WRITE,COPY,ADD,TRIAD,QUADAD,PENTAD,HEXAD : STREAM type: One can remove any of these stream, but the order should be maintaned as of now.
MEMORYMODE = NONUM                                     : Memory models , NONUM = Zero-Copy , UM = Unified Memory
REPETATIONS = 1                                        : # of Repeatations for the experiments to avoid measurement noise
LATENCY = Y                                            : Latency experiements will take place if "Y" else no latency experiements. Both THROUGHPUT and LATENCY flag can not be "Y" at the same time.
ITERATIONS = 1                                         : # of iterations for pointer chasing, kept 1 by default for our experiments
STRIDE = 4B-512K,2                                     : Starting minimum stride size - Last Maximum stride size, Incrementing value, Incrementing operator ( * - multiplications, + - Addition )
CHASE_ELEMENTS = 128-16384,2,*                         : Starting minimum # of elements chased - Last Maximum # of elements chased, Incrementing value, Incrementing operator ( * - multiplications, + - Addition )
START_THREAD = 1024     							   : Starting gpu block size
END_THREAD = 1024                                      : Maximum GPU block size
OMP_THREADS = 1-1,4                                    : Starting minimum Openmp thread - Last Maximum Openmp threads, Incrementing value to be multiplied
MEM_ADVISE = N                                         : cuMemAdvise() experiment is performed if "Y" with MEMORYMODE = UM , if "N" , it will not take place. 
THREAD_EXPERIMENT = N                                  : Reduction of Redudant page fault experiments will take place if "Y" with MEMORYMODE = UM else no.

Execution:

Execution can be done executing simple make command after correct configuration. Different make commands are,

1. make nvl0 : Complete compilation and execution of the experiments for all the range of data size given in configuration file for NVLINK system.
2. make pcie0 : Complete compilation and execution of the experiments for all the range of data size given in configuration file for PCIe system.
3. make cmp0: Only compilation for NVLINK System.
4. make cmp1: Only compilation for PCIe system.

For individual execution, one needs to run following command,

./nvprog <A> <B> <C> <D>: 

A : Data size for throughput / # of chasing elements for latency
B : Stride length in elements
C : # of Open mp threads
D : 1 for throughput , 0 for latency 

Output:

After execution, output will be stored inside /data/ folder as csv and text file with following naming conventions.

1. result-pchase-lat-nonum.txt or result-pchase-lat-nonum.csv - for zero copy latency experiment.
2. result-pchase-lat-um.txt or result-pchase-lat-um.csv - for Unified memory atency experiments.
3. result-stream-bw-nonmum.txt or result-stream-bw-nonmum.csv - zero-copy memory throughput output.
4. result-stream-bw-um.txt or result-stream-bw-um.csv - Unified memory throughput output.
5. result-stream-bw-memcpy.txt or result-stream-bw-memcpy.csv - cuda mem copy output.

For complete execution, data file from previous executions are always backed up.
