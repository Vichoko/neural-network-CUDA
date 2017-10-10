CPP = g++
CPPFLAGS = -g -c

NVCC = /usr/local/cuda-8.0/bin/nvcc
FINALCUDAFLAGS = --cudart static --relocatable-device-code=false -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50 -link
OBJCUDAFLAGS = -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50  -x cu

all: main


main: src/neural-network.o src/parser.o
	$(NVCC) $(FINALCUDAFLAGS) -o main src/neural-network.o src/parser.o  

src/neural-network.o : src/neural-network.cu
	$(NVCC)  $(OBJCUDAFLAGS) -o src/neural-network.o src/neural-network.cu

src/parser.o : src/resources/parser.cpp
	$(NVCC) $(CPPFLAGS) -o src/parser.o src/resources/parser.cpp

clean:
	rm -rf *.o src/*.o main




