CC=cc
CFLAGS= 
LDFLAGS=
SOURCE2=execute.c
OBJECTS=$(SOURCE1:.c=.o)
EXECUTABLE=stream
COMMON=

all: stream

stream:
	$(CC) $(CFLAGS) $(SOURCE1) -o stream
	
prof:	
	nvprof --print-gpu-trace ./nvprog 33554432

nvl0:
	nvcc ./config/main.cu ./config/config.cu ./config/log.cu -o param
	./param
	nvcc -c main.cpp CUDAStream.cu CUDAPtrchase.cu -m64 -arch=sm_60 -Xcompiler -O0 -Xcompiler -fopenmp --std=c++11 -ptx -lnvidia-m -lnvidia-ml
	nvcc -c main.cpp CUDAStream.cu CUDAPtrchase.cu -m64 -arch=sm_60 -Xcompiler -O0 -Xcompiler -fopenmp --std=c++11 -Xptxas -dlcm=cg -lnvidia-ml
	nvcc -link -lcuda main.o CUDAStream.o CUDAPtrchase.o -m64 -arch=sm_60 -Xcompiler -O0 -Xcompiler -fopenmp --std=c++11 -o nvprog -Xptxas -dlcm=cg -lnvidia-ml
	g++ $(SOURCE2) -o ex
	./ex 1
	chmod 777 execute.sh
	./execute.sh
cmp0:
	nvcc ./config/main.cu ./config/config.cu ./config/log.cu -o param
	./param
	nvcc -c main.cpp CUDAStream.cu CUDAPtrchase.cu -m64 -arch=sm_60 -Xcompiler -O0 -Xcompiler -fopenmp -Xcompiler -s --std=c++11 -lnvidia-m -lnvidia-ml
	nvcc -c main.cpp CUDAStream.cu CUDAPtrchase.cu -m64 -arch=sm_60 -Xcompiler -O0 -Xcompiler -fopenmp --std=c++11 -Xptxas -dlcm=cg -lnvidia-ml
	nvcc -link -lcuda main.o CUDAStream.o CUDAPtrchase.o -m64 -arch=sm_60 -Xcompiler -O0 -Xcompiler -fopenmp --std=c++11 -o nvprog -Xptxas -dlcm=cg -lnvidia-ml
	
cmp1:
	nvcc ./config/main.cu ./config/config.cu ./config/log.cu -o param
	./param
	nvcc -c main.cpp CUDAStream.cu CUDAPtrchase.cu -m64 -arch=sm_37 -Xcompiler -O0 -Xcompiler -fopenmp -Xcompiler -s --std=c++11 -lnvidia-m -lnvidia-ml
	nvcc -c main.cpp CUDAStream.cu CUDAPtrchase.cu -m64 -arch=sm_37 -Xcompiler -O0 -Xcompiler -fopenmp --std=c++11 -Xptxas -dlcm=cg -lnvidia-ml
	nvcc -link -lcuda main.o CUDAStream.o CUDAPtrchase.o -m64 -arch=sm_37 -Xcompiler -O0 -Xcompiler -fopenmp --std=c++11 -o nvprog -Xptxas -dlcm=cg -lnvidia-ml

pcie0:
	nvcc ./config/main.cu ./config/config.cu ./config/log.cu -o param
	./param
	nvcc -c main.cpp CUDAStream.cu CUDAPtrchase.cu -m64 -arch=sm_37 -Xcompiler -O0 -Xcompiler -fopenmp --std=c++11 -ptx -lnvidia-m -lnvidia-ml
	nvcc -c main.cpp CUDAStream.cu CUDAPtrchase.cu -m64 -arch=sm_37 -Xcompiler -O0 -Xcompiler -fopenmp --std=c++11 -Xptxas -dlcm=ca -lnvidia-ml
	nvcc -link -lcuda main.o CUDAStream.o CUDAPtrchase.o -m64 -arch=sm_37 -Xcompiler -O0 -Xcompiler -fopenmp --std=c++11 -o nvprog -Xptxas -dlcm=ca -lnvidia-ml
	g++ $(SOURCE2) -o ex
	./ex 1
	chmod 777 execute.sh
	./execute.sh

ex0: 
	./ex
	chmod 777 execute.sh
	./execute.sh

clean:
	
.PHONY : all clean
