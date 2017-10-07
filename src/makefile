COMPILER= g++
CFLAGS = -ggdb -Wall -Wno-unused-variable -Wno-unused-but-set-variable -lrt -O3
OPENCL = -I/usr/include/ -L/usr/lib64/ -Wl,-rpath,/usr/lib64/ -L/usr/lib/nvidia-319-updates/ -I/opt/AMDAPP/include -L/opt/AMDAPP/lib/x86_64
LDFLAGS = -lOpenCL -Wl,-rpath=/usr/local/cuda-8.0/lib64 # add -Wl,rpath=PATH option so that  PATH is searched for shared libraries -i.e. libcudart.su.5.5
CUDA_FLAGS = -L/usr/local/cuda-8.0/lib64 -lcudart -lcuda

SOURCES = examples.cpp

CUDA_SOURCES = neural-network.cu

OBJECTS = $(SOURCES:.c=.o) $(CUDA_SOURCES:.cu=.o)
EXECUTABLE = test


all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE) : $(OBJECTS)
	$(COMPILER) $(OBJECTS) $(OPENCL) $(CUDA_FLAGS) $(LDFLAGS) -o $@

# build all the object files
examples.o : examples.cpp
	/usr/local/cuda-8.0/bin/nvcc $(CUDA_FLAGS) -m64 -arch=sm_35 -o examples.o --shared -Xcompiler -fPIC -c examples.cpp
	#$(COMPILER) $(CFLAGS) examples.cpp

#aes.o: aes.c
#	$(g++) $(CFLAGS) aes.c

#aes.o: aes.c
#	/usr/local/cuda-8.0/bin/nvcc $(CUDA_FLAGS) -m64 -arch=sm_35 -o aes.o --shared -Xcompiler -fPIC -c aes.c

#expand_key.o: expand_key.c
#	$(g++) $(CFLAGS) expand_key.c

#cpu_cipher.o: cpu_cipher.c
#	$(g++) $(CFLAGS) cpu_cipher.c

#opencl_cipher.o: opencl_cipher.c
#	$(g++) $(CFLAGS) $(OPENCL) opencl_cipher.c

neural-network.o: neural-network.cu
	/usr/local/cuda-8.0/bin/nvcc $(CUDA_FLAGS) -m64 -arch=sm_35 -o neural-network.o --shared -Xcompiler -fPIC -c neural-network.cu
#/usr/local/cuda-8.0/bin/nvcc $(CUDA_FLAGS) -m64 -arch=sm_20 -arch=sm_35 -o cuda_cipher.o --shared -Xcompiler -fPIC -c cuda_cipher.cu

# do cleanup - remove object files and executable
clean:
	rm -rf $(OBJECTS) $(EXECUTABLE)
