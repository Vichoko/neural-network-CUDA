COMPILER= g++
CFLAGS = -ggdb -Wall -Wno-unused-variable -Wno-unused-but-set-variable -lrt -O3
OPENCL = -I/usr/include/ -L/usr/lib64/ -Wl,-rpath,/usr/lib64/ -L/usr/lib/nvidia-319-updates/ -I/opt/AMDAPP/include -L/opt/AMDAPP/lib/x86_64
LDFLAGS = -lOpenCL -Wl,-rpath=/usr/local/cuda-8.0/lib64 # add -Wl,rpath=PATH option so that  PATH is searched for shared libraries -i.e. libcudart.su.5.5
CUDA_FLAGS = -L/usr/local/cuda-8.0/lib64 -lcudart -lcuda

SOURCES = examples.cpp

CUDA_SOURCES = neural-network.cu

OBJECTS = $(SOURCES:.c=.o) $(CUDA_SOURCES:.cu=.o)
EXECUTABLE = test

NVCC = /usr/local/cuda-8.0/bin/nvcc


all: main


main: neural-network.o 
	/usr/local/cuda-8.0/bin/nvcc --cudart static --relocatable-device-code=false -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50 -link -o  main  neural-network.o   

#neural-network.d : neural-network.cu
#	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 -gencode arch=compute_50,code=sm_50  -M -o neural-network.d neural-network.cu

neural-network.o : neural-network.cu
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50  -x cu -o  neural-network.o neural-network.cu



clean:
	rm -rf *.o main

