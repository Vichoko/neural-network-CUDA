
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <signal.h>

// public functions
#define LEN(x)  (sizeof(x) / sizeof((x)[0]))


int init_network_struct(int input_size);
// entrenar
int train(double** input, double** expected_output, int n_epochs);

// MAIN
int main();

/** tests **/
// compuertas logicas
__host__ int networkLogicGateSingleOutputTest(double** input, double** expected_output, int data_count, bool verbose);
int networkORSingleOutputLearningTest(bool verbose);
int networkANDSingleOutputLearningTest(bool verbose);

/** control functions **/
float train(double** input, double** expected_output, int data_count, 
	int n_epochs, bool run_in_parallel, bool verbose);


__host__ void sequential_forward_feed(double* input);
__host__ void parallel_forward_feed(double* input);

__host__ void sequential_back_propagation(double* expected_output, bool verbose);
__host__ void parallel_back_propagation(double* expected_output, int expected_output_len, bool verbose);

__host__ void parallel_update_weights(double* input);
__host__ int sequential_update_weights(double* input);


/** kernels **/
__global__ void first_layer_synapsis(double* input, double** weights, double* biases, double* last_outputs);
__global__ void non_first_layer_synapsis(int* layer_index, double** weights,double* biases, double* last_outputs);

__global__ void last_layer_back_propagation(double* last_outputs, double* deltas, double* expected_output);
__global__ void non_last_layer_back_propagation(int* layer_index_ptr, double* last_outputs, double** weights, double* deltas);

__global__ void update_weights(double* input, double* deltas, double** weights, double* biases, double* last_outputs);


/** util functions **/
__host__ __device__ int layer_neuron_map(int layer_index, int neuron_index);
__host__ __device__ double sigmoid(double input);
__host__ __device__ double transfer_derivative(double output);
__device__ double atomicDoubleAdd(double* address, double val);

// SIGINT HANDLER
void intHandler(int dummy);

/** memory management **/
__host__ int init_network_struct(int input_size, bool verbose);
int destroy_net();


__host__ int free_weights_in_gpu();
__host__ int alloc_weights_in_gpu();
__host__ int copy_weights_to_gpu(bool is_update);
__host__ int copy_weights_to_host();
__host__ int copy_biases_to_gpu();
__host__ int copy_biases_to_host();
