#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <signal.h>
#include <vector>

#include "neural-network.h"
#include "neural-network.cuh"
#include "resources/parser.h"

// GLOBALS
// device memory pointers
double* d_biases;
double* d_deltas;
double* d_last_outputs;
double** d_weights;
//int* d_neuron_counts_array;
void** temp_d_ptrs;

// host memory pointers
double* biases;
double* deltas;
double* last_outputs;
double** weights;
//int* neuron_counts_array;

#define THREADS_PER_BLOCK 16 // tiene que ser potencia de 2 y mayor o igual que la cantidad maxima de neuronas que puede haber por capa
#define INPUT_SIZE 2
#define NEURONS_QUANTITY 2
#define LAYERS_QUANTITY 2
#define LEARNING_RATE 0.1

// metricas
int gpu_mode;
float seconds_feeding;
float seconds_updating_weights;
float seconds_back_propagating;
unsigned int neurons_feeded;
unsigned int neurons_back_propagated;
unsigned int neurons_updated_weights;

// int main() {

// }

int main()
{
	signal(SIGINT, intHandler);
 	networkANDSingleOutputLearningTest(true);
 	intHandler(0);


/* Clasificador de texto: Lanza segmentation fault
	init_network_struct(INPUT_SIZE, true);
	raw_data_container_t* reti = parse("spam.csv", true);
	data_container_t* ret = preprocess_texts(reti);
	double** ham_bag_of_words = ret->ham_bag;
	double** spam_bag_of_words = ret->spam_bag;
	int ham_data_len = ret->ham_data_len;
	int spam_data_len = ret->spam_data_len;
	int data_len = ham_data_len + spam_data_len;


	double** input = (double**) malloc(sizeof(double*) * data_len);
	double** output = (double**) malloc(sizeof(double*) * data_len);

	// porque la red neuronal iiene cantidad fija de neuronas, el output tiene la misma cantdidad que capa de entrada; i.e. dict_size
	double* spam_output = (double*) malloc(sizeof(double) * ret->dict_size);
	double* ham_output = (double*) malloc(sizeof(double) * ret->dict_size);
	for (int i = 0; i < ret->dict_size; i++){
		spam_output[i] = 1;
		ham_output[i] = 0;
	}

	int data_counter = 0;
	int ham_counter = 0;
	int spam_counter = 0;
	while(data_counter <= data_len){
		if (ham_counter <= ham_data_len){
			input[data_counter] = ham_bag_of_words[ham_counter];
			output[data_counter] = ham_output;
			data_counter++;
			ham_counter++;
		}
		if (spam_counter <= spam_data_len){
			input[data_counter] = spam_bag_of_words[spam_counter];
			output[data_counter] = spam_output;
			data_counter++;
			spam_counter++;
		}
	}

	// Tirar todo
	float a = train(input, output, data_len, 100000, false, true);
	printf("finish all\n");
	free(ret);
	free(spam_output);
	free(ham_output);
	free(output);
	for (int i = 0; i < data_len; i++){
		free(input[i]);
	}
	free(input);
	*/
	return 0;
}


float train(double** input, double** expected_output, int data_count, int n_epochs, bool run_in_parallel, bool verbose) {
	gpu_mode = run_in_parallel;
	if (run_in_parallel){
		printf("Training in GPU\n");
		} else {
			printf("Training in CPU\n");
		}
	printf("Finish with CTRL+C to see stats.\n");	
		
	int output_size = NEURONS_QUANTITY; //todo: variable
	// recibe dataset de entrenamiento; varios input con sus respectivos output
	if (sizeof(input) != sizeof(expected_output)) {
		printf("train :: dataset input and expectedOutput arrays have different lenghts: %lu and %lu.\n",
				sizeof(input), sizeof(expected_output));
		exit(-1);
	}
	printf("learnRate: %f\n", LEARNING_RATE);
	printf("numer of epochs: %d\n", n_epochs);
	//double errors[nEpochs];
	// Siguientes variables son para no saturar de impresiones en consola.
	int epochsPart = n_epochs / 1000;
	int counter = 0;
	for (int epoch_index = 0; epoch_index < n_epochs; epoch_index++) {
		// epochs se ejecutan en secuencia. No paralelizable (se necesita resultado previo para el siguiente).
		double sumError = 0;
		for (int data_index = 0; data_index < data_count; data_index++) {
			// entrenar sobre cada par de vectores input/output. No paralelizable (se necesita exclusión mutua durante el proeceso).

			// FORWARD FEEDING
			clock_t start = clock();
			if (run_in_parallel){
				parallel_forward_feed(input[data_index]); 
				//sequential_forward_feed(input[data_index]); // todo: implement sequential

			} else {
				sequential_forward_feed(input[data_index]); // todo: implement sequential
			}
			clock_t end = clock();
			seconds_feeding += (float)(end - start) / CLOCKS_PER_SEC;
			neurons_feeded += LAYERS_QUANTITY * NEURONS_QUANTITY;

			// ERROR RECORDING
			// todo: mover cudamemcpy de outputs aqui, para no afectar mediciones
			if (run_in_parallel){
				cudaError_t err = cudaSuccess;
				cudaMemcpy(last_outputs, d_last_outputs,
				sizeof(double) * NEURONS_QUANTITY * LAYERS_QUANTITY,
				cudaMemcpyDeviceToHost);
				cudaDeviceSynchronize();
				err = cudaGetLastError();
				if (err != cudaSuccess) {
					printf("error bringing last_outputs: %s\n", cudaGetErrorString(err));
				}


			}
			double* actual_output = &last_outputs[layer_neuron_map(LAYERS_QUANTITY - 1, 0)];
			if (sizeof(actual_output) != sizeof(expected_output[data_index])) {
				printf("train :: one of layers actualOutput/expectedOutput have different sizes.\n");
				exit(-2);
			}
			for (int outputIndex = 0; outputIndex < output_size;
					outputIndex++) {
				// Para cada input se calcula el error cuadratico medio para visualizar aprendizaje
				sumError += (expected_output[data_index][outputIndex] - actual_output[outputIndex])
						* (expected_output[data_index][outputIndex] - actual_output[outputIndex]);

			}
			// BACK PROPAGATION
			start = clock();
			if (run_in_parallel){
				//sequential_back_propagation(expected_output[data_index], false); //aqui esta el sapo
				parallel_back_propagation(expected_output[data_index], LEN(expected_output[data_index]), false); // todo: implement parallel
			} else {
				sequential_back_propagation(expected_output[data_index], false);
			}
			end = clock();
			seconds_back_propagating += (float)(end - start) / CLOCKS_PER_SEC;
			neurons_back_propagated += LAYERS_QUANTITY * NEURONS_QUANTITY;

			// UPDATE WEIGHTS
			start = clock();
			if (run_in_parallel){
				//sequential_update_weights(input[data_index]);
				parallel_update_weights(input[data_index]);
			} else {
				sequential_update_weights(input[data_index]);
			}
			end = clock();
			seconds_updating_weights += (float)(end - start) / CLOCKS_PER_SEC;
			neurons_updated_weights += LAYERS_QUANTITY * NEURONS_QUANTITY;

		}
		if (++counter >= epochsPart) {
			printf("Epoch: %d , error: %f\n", epoch_index, sumError);
			counter = 0;
		}
	}
	return 0;
}


/*****************************/
/** TESTS **/
/*****************************/

__host__ int networkLogicGateSingleOutputTest(double** input, double** expected_output, int data_count, bool verbose) {
	init_network_struct(INPUT_SIZE, false);

	/** Red con 2 neuronas de entrada, 2 escondidas y una de salida, generada con pesos aleatorios, para aprender XOR.
	 * Clases binarias:
	 * 	1, si bit_1 <GATE> bit_2 == 1;
	 * 	0 si no.*/

	if (verbose) {
		printf("initializing training.\n");
	}
	bool run_in_parallel = false;
	train(input, expected_output, data_count, 1000000, run_in_parallel, verbose);
	print_stats();
	
	run_in_parallel = true;
	train(input, expected_output, data_count, 1000000, run_in_parallel, verbose);

	if (verbose) {
		printf("	done training.\n");
	}

	/*	// preparar datos de prueba
	 int casosTotales = 300000;
	 double* testInput[casosTotales];
	 double* testOutput[casosTotales];
	 for (int i = 0; i < casosTotales; i++) {
	 double seed = rand() % 1000 / 1000;

	 int randomIndex;
	 if (seed < 0.25)
	 randomIndex = 0;
	 else if (seed < 0.5)
	 randomIndex = 1;
	 else if (seed < 0.75)
	 randomIndex = 2;
	 else
	 randomIndex = 3;
	 testInput[i] = input[randomIndex];
	 testOutput[i] = expected_output[randomIndex];
	 }

	 double threshold = 0.5;*/
	destroy_net();
	return 1;
}

int networkANDSingleOutputLearningTest(bool verbose) {
// Pocas combinaciones posibles, mas enfasis al numero de epochs

	double* in_one = (double*) malloc(sizeof(double) * 2);
	double* in_two = (double*) malloc(sizeof(double) * 2);
	double* in_three = (double*) malloc(sizeof(double) * 2);
	double* in_four = (double*) malloc(sizeof(double) * 2);

	in_one[0] = 0;
	in_one[1] = 0;
	in_two[0] = 0;
	in_two[1] = 1;
	in_three[0] = 1;
	in_three[1] = 0;
	in_four[0] = 1;
	in_four[1] = 1;

	double** input = (double**) malloc(sizeof(double*) * 4);
	input[0] = in_one;
	input[1] = in_two;
	input[2] = in_three;
	input[3] = in_four;

	double* out_one = (double*) malloc(sizeof(double));
	double* out_two = (double*) malloc(sizeof(double));
	double* out_three = (double*) malloc(sizeof(double));
	double* out_four = (double*) malloc(sizeof(double));

	*out_one = 0;
	*out_two = 0;
	*out_three = 0;
	*out_four = 1;

	double** expectedOutput = (double**) malloc(sizeof(double*) * 4);
	expectedOutput[0] = out_one;
	expectedOutput[1] = out_two;
	expectedOutput[2] = out_three;
	expectedOutput[3] = out_four;
	if (verbose) {
		printf("expectedOutput and expectedOutput generated for AND test.\n");
	}

	networkLogicGateSingleOutputTest(input, expectedOutput, 4, verbose);
	free(out_one);
	free(out_two);
	free(out_three);
	free(out_four);
	free(in_one);
	free(in_two);
	free(in_three);
	free(in_four);
	return 1;
}
int networkORSingleOutputLearningTest(bool verbose) {
// Pocas combinaciones posibles, mas enfasis al numero de epochs

	double* in_one = (double*) malloc(sizeof(double) * 2);
	double* in_two = (double*) malloc(sizeof(double) * 2);
	double* in_three = (double*) malloc(sizeof(double) * 2);
	double* in_four = (double*) malloc(sizeof(double) * 2);

	in_one[0] = 0;
	in_one[1] = 0;
	in_two[0] = 0;
	in_two[1] = 1;
	in_three[0] = 1;
	in_three[1] = 0;
	in_four[0] = 1;
	in_four[1] = 1;

	double** input = (double**) malloc(sizeof(double*) * 4);
	input[0] = in_one;
	input[1] = in_two;
	input[2] = in_three;
	input[3] = in_four;

	double* out_one = (double*) malloc(sizeof(double));
	double* out_two = (double*) malloc(sizeof(double));
	double* out_three = (double*) malloc(sizeof(double));
	double* out_four = (double*) malloc(sizeof(double));

	*out_one = 0;
	*out_two = 1;
	*out_three = 1;
	*out_four = 1;

	double** expectedOutput = (double**) malloc(sizeof(double*) * 4);
	expectedOutput[0] = out_one;
	expectedOutput[1] = out_two;
	expectedOutput[2] = out_three;
	expectedOutput[3] = out_four;
	if (verbose) {
		printf("expectedOutput and expectedOutput generated for AND test.\n");
	}

	networkLogicGateSingleOutputTest(input, expectedOutput, 4, verbose);
	free(out_one);
	free(out_two);
	free(out_three);
	free(out_four);
	free(in_one);
	free(in_two);
	free(in_three);
	free(in_four);
	return 1;
}


/*****************************/
/** FORWARD FEED **//** FORWARD FEED **//** FORWARD FEED **/
/*****************************/

__global__ void first_layer_synapsis(double* input, double** weights, double* biases, double* last_outputs) {
	//__shared__ int ponderated_sum_results[THREADS_PER_BLOCK];
	__shared__ double acc;
	if (threadIdx.x == 0){
		acc = 0;
	}

	int input_index = threadIdx.x;
	int neuron_index = blockIdx.x;

	if (neuron_index < NEURONS_QUANTITY) {
		// un bloque por neurona
		if (input_index < INPUT_SIZE) {
			// un thread por input/
			// cada thread calcula una suma ponderada
			atomicDoubleAdd(&acc,
					weights[layer_neuron_map(0, neuron_index)][input_index] * input[input_index]);
			__syncthreads(); // esperar a que se calcule suma ponderada todos los thread

			if (threadIdx.x == 0){
				double result = acc
						+ biases[layer_neuron_map(0, neuron_index)];
				last_outputs[layer_neuron_map(0, neuron_index)] = sigmoid(result);
			}
		}
	}
}
__global__ void non_first_layer_synapsis(int* layer_index, double** weights,double* biases, double* last_outputs) {
	__shared__ double acc;
	if (threadIdx.x == 0){
		acc = 0;
	}
	int input_index = threadIdx.x;
	int neuron_index = blockIdx.x;

	if (neuron_index < NEURONS_QUANTITY) {
		// un bloque por neurona
		if (input_index < NEURONS_QUANTITY) {
			// un thread por input
			atomicDoubleAdd(&acc, weights[layer_neuron_map(*layer_index, neuron_index)][input_index] *
					last_outputs[layer_neuron_map((*layer_index) - 1, input_index)]);
			__syncthreads(); // esperar a que se calcule suma ponderada todos los thread

			if (threadIdx.x == 0){
				//printf("peso de layer 0, neurona 0, input 0 es %f\n", weights[layer_neuron_map(0, 0)][0]);
				double result = acc
						+ biases[layer_neuron_map(*layer_index, neuron_index)];
				last_outputs[layer_neuron_map(*layer_index, neuron_index)] = sigmoid(result);
			}

		}
	}
}
__host__ void parallel_forward_feed(double* input) {
	//No completamente paralelizable (se necesita resultado de capa previa para la siguiente).
	// Cada capa es paralelizable, se debe calcular producto punto entre pesos de la layer e inputs, luego sumar bias.

	// un bloque por neurona, un thread por peso/input, computa su suma ponderada
	double* d_input;
	int* d_layer_index;

	cudaMalloc((void **) &d_input, sizeof(double) * INPUT_SIZE);
	cudaMalloc((void **) &d_layer_index, sizeof(int));

	// necesary
	cudaMemcpy(d_input, input, sizeof(double) * INPUT_SIZE,
			cudaMemcpyHostToDevice);

	// todo: delete, maybe is unnecesary to do if everyting is parallel
	//copy_weights_to_gpu(true);
	//copy_biases_to_gpu();

	// caso base
	first_layer_synapsis<<<
			NEURONS_QUANTITY,
			THREADS_PER_BLOCK>>>(d_input, d_weights, d_biases,
			d_last_outputs);

	cudaDeviceSynchronize(); // esperar que termine primera capa

	// caso iterativo
	for (int layer_index = 1; layer_index < LAYERS_QUANTITY; layer_index++) {
		cudaMemcpy(
			d_layer_index, 
			&layer_index, 
			sizeof(int),
			cudaMemcpyHostToDevice);
		// actualizar layer_index

		non_first_layer_synapsis<<<NEURONS_QUANTITY, THREADS_PER_BLOCK>>>(d_layer_index, d_weights, d_biases,
				d_last_outputs);
		cudaDeviceSynchronize(); // esperar que termine segunda capa
	}

	// bring result from device to host
	// todo: maybe is unnecesary if everything is parallel


	// free auxiliary data
	cudaFree(d_input);
	cudaFree(d_layer_index);
	return;
}
__host__ void sequential_forward_feed(double* input){
	// CASO BASE: Capa de input
	for (int neuron_index = 0; neuron_index < NEURONS_QUANTITY; neuron_index++){
		double acc = 0;
		for (int input_index = 0; input_index < INPUT_SIZE; input_index++){
			acc += weights[layer_neuron_map(0, neuron_index)][input_index] * input[input_index];
		}
		acc += biases[layer_neuron_map(0, neuron_index)];
		last_outputs[layer_neuron_map(0, neuron_index)] = sigmoid(acc);
	}	

	// CASO ITERATIVO: Resto de las capas
	for (int layer_index = 1; layer_index < LAYERS_QUANTITY; layer_index++){
		for (int neuron_index = 0; neuron_index < NEURONS_QUANTITY; neuron_index++){
			double acc = 0;
			for (int input_index = 0; input_index < NEURONS_QUANTITY; input_index++){
				acc += weights[layer_neuron_map(layer_index - 1, neuron_index)][input_index] * last_outputs[layer_neuron_map(layer_index - 1, input_index)];
			}
			acc += biases[layer_neuron_map(layer_index, neuron_index)];
			last_outputs[layer_neuron_map(layer_index, neuron_index)] = sigmoid(acc);
		}
	}
}

/*****************************/
/** BACK PROPAGATION **//** BACK PROPAGATION **//** BACK PROPAGATION **/
/*****************************/

__global__ void last_layer_back_propagation(double* last_outputs, double* deltas, double* expected_output){
	int neuron_index = threadIdx.x + blockIdx.x * blockDim.x;
	int layer_index = LAYERS_QUANTITY-1;

	if (neuron_index < NEURONS_QUANTITY){
		double actual_output = last_outputs[layer_neuron_map(layer_index, neuron_index)];
		deltas[layer_neuron_map(layer_index, neuron_index)] = (expected_output[neuron_index] - actual_output)* transfer_derivative(actual_output);
	}
}


__global__ void non_last_layer_back_propagation(int* layer_index_ptr, double* last_outputs, double** weights, double* deltas){
	int layer_index = *layer_index_ptr;
	__shared__ double acc;
	if (threadIdx.x == 0){
		acc = 0;
	}

	int neuron_index = blockIdx.x;
	int neighbor_neuron_index = threadIdx.x;

	if (neuron_index < NEURONS_QUANTITY){
		double actual_output = last_outputs[layer_neuron_map(layer_index, neuron_index)];

		if (neighbor_neuron_index < NEURONS_QUANTITY){
			// un bloque por neurona, un thread por neurona vecina
			atomicDoubleAdd(&acc, weights[layer_neuron_map(layer_index + 1, neighbor_neuron_index)][neuron_index] * deltas[layer_neuron_map(layer_index + 1, neighbor_neuron_index)]);
		}
		__syncthreads(); // esperar a que se calcule suma ponderada todos los thread
		deltas[layer_neuron_map(layer_index, neuron_index)] = acc*transfer_derivative(actual_output);

	}
}

/*
__global__ void non_last_layer_back_propagation(int* layer_index_ptr, double* last_outputs, double** weights, double* deltas){
	int neuron_index = threadIdx.x + blockIdx.x * blockDim.x;

	int layer_index = *layer_index_ptr;
	double acc = 0;

	if (neuron_index < NEURONS_QUANTITY){
		double actual_output = last_outputs[layer_neuron_map(layer_index, neuron_index)];

		for (int neighbor_neuron_index = 0; neighbor_neuron_index < NEURONS_QUANTITY; neighbor_neuron_index++){
			acc += weights[layer_neuron_map(layer_index + 1, neighbor_neuron_index)][neuron_index] * deltas[layer_neuron_map(layer_index + 1, neighbor_neuron_index)];
		}

		deltas[layer_neuron_map(layer_index, neuron_index)] = acc*transfer_derivative(actual_output);

	}
}
*/
__host__ void parallel_back_propagation(double* expected_output, int expected_output_len, bool verbose){
	// partiendo de layer de salida, un thread por neurona, actualizar delta
	double* d_expected_output;
	int* d_layer_index;

	cudaMalloc((void **) &d_expected_output, sizeof(double) * expected_output_len);
	cudaMalloc((void **) &d_layer_index, sizeof(int));

	// necesary
	cudaMemcpy(
		d_expected_output, 
		expected_output, 
		sizeof(double) * expected_output_len, 
		cudaMemcpyHostToDevice);
	//cudaMemcpy(d_last_outputs, last_outputs, sizeof(double) * NEURONS_QUANTITY * LAYERS_QUANTITY, cudaMemcpyHostToDevice);
	//copy_weights_to_gpu(true);


	// d_layer_index, d_last_outputs, d_weights, d_deltas already in GPU memory
	last_layer_back_propagation<<<NEURONS_QUANTITY, THREADS_PER_BLOCK>>>(d_last_outputs, d_deltas, d_expected_output);
	for (int layer_index = LAYERS_QUANTITY - 2; layer_index >= 0; layer_index--){
		cudaDeviceSynchronize();
		cudaMemcpy(d_layer_index, 
			&layer_index, 
			sizeof(int), 
			cudaMemcpyHostToDevice);
		non_last_layer_back_propagation<<<NEURONS_QUANTITY, THREADS_PER_BLOCK>>>(d_layer_index, d_last_outputs, d_weights, d_deltas);
	}

	// bring result from device to host
	// todo: maybe is unnecesary if everything is parallel
	//cudaMemcpy(deltas, d_deltas, sizeof(double) * NEURONS_QUANTITY * LAYERS_QUANTITY, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	// free auxiliary data
	cudaFree(d_expected_output);
	cudaFree(d_layer_index);
	return;
} 
__host__ void sequential_back_propagation(double* expected_output, bool verbose) {
	for (int layer_index = LAYERS_QUANTITY - 1; layer_index >= 0;
			layer_index--) {
		if (verbose)
			printf("back propagating layer: %d\n", layer_index);
		// backward iteration. No paralelizable (se necesita resultado previo para siguiente capa).
		if (layer_index == LAYERS_QUANTITY - 1) {
			// Caso capa de salida

			for (int neuron_index = 0; neuron_index < NEURONS_QUANTITY;
					neuron_index++) {
				// todo: paralelizar este for
				// ultimo output generado por esta neurona
				double actual_output = last_outputs[layer_neuron_map(
						layer_index, neuron_index)];

				// actualizar delta en base a actual versus expected.
				deltas[layer_neuron_map(layer_index, neuron_index)] =
						(expected_output[neuron_index] - actual_output)
								* transfer_derivative(actual_output);
			}
		} else {
			// caso capa escondida o de entrada
			for (int neuron_index = 0; neuron_index < NEURONS_QUANTITY;
					neuron_index++) {
				// por cada neurona de la capa
				// todo: paralelizar este for

				// ultimo output generado por esta neurona

				double actual_output = last_outputs[layer_neuron_map(
						layer_index, neuron_index)];

				// transfiero el error de las neuronas que la suceden
				double error = 0;
				for (int neighbor_neuron_index = 0;
						neighbor_neuron_index < NEURONS_QUANTITY;
						neighbor_neuron_index++) {

					// por cada neurona que le llega su output
					// peso de neurona vecina con output de neurona

					if (weights[layer_neuron_map(layer_index + 1,
							neighbor_neuron_index)] == NULL) {
						printf("neighbor layer weigth are null.\n");
						exit(-3);
					}
					double weight = weights[layer_neuron_map(layer_index + 1,
							neighbor_neuron_index)][neuron_index];

					// delta de neurona vecina
					double delta = deltas[layer_neuron_map(layer_index + 1,
							neighbor_neuron_index)];

					error += (weight * delta);

				}
				// actualizar delta de neurona
				deltas[layer_neuron_map(layer_index, neuron_index)] = error
						* transfer_derivative(actual_output);
			}
		}
	}
	return;
}


/*****************************/
/** UPDATE WEIGHTS **//** UPDATE WEIGHTS **//** UPDATE WEIGHTS **/
/*****************************/
__global__ void update_weights(double* input, double* deltas, double** weights, double* biases, double* last_outputs){
	// 1 thread por peso/input, 1 bloque por neurona
	int input_size = INPUT_SIZE;
	int input_index = threadIdx.x;
	int layer_index = blockIdx.x % NEURONS_QUANTITY;
	int neuron_index = blockIdx.x - NEURONS_QUANTITY * layer_index;


	if (layer_index > 0){
		// Si no es input layer, los input vienen de layers previas
		input = &last_outputs[layer_neuron_map(layer_index - 1, 0)];// direccion de primer elemento de la layer
		input_size = NEURONS_QUANTITY;
	}

	if (neuron_index < NEURONS_QUANTITY && input_index < input_size && layer_index < LAYERS_QUANTITY){
		// un bloque por neurona, un thread por input/peso
		double real_delta = (LEARNING_RATE * deltas[layer_neuron_map(layer_index, neuron_index)]);
		weights[layer_neuron_map(layer_index, neuron_index)][input_index] += real_delta * input[input_index];
		
		if (neuron_index == 0){
			// un thread encargado de actualizar bias
			//printf("peso updateado de layer 0, neurona 0, input 0 era %f queda en %f\n", weights[layer_neuron_map(0, 0)][0] - real_delta * input[0], weights[layer_neuron_map(0, 0)][0]);
			biases[layer_neuron_map(layer_index, neuron_index)] += real_delta;
		}
	}
}
__host__ void parallel_update_weights(double* input){
	double* d_input;

	cudaMalloc((void **) &d_input, sizeof(double) * INPUT_SIZE);

	// necesary
	cudaMemcpy(d_input, input, sizeof(double) * INPUT_SIZE,
			cudaMemcpyHostToDevice);
	//cudaMemcpy(d_deltas, deltas, sizeof(double) * NEURONS_QUANTITY * LAYERS_QUANTITY, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_last_outputs, last_outputs, sizeof(double) * NEURONS_QUANTITY * LAYERS_QUANTITY, cudaMemcpyHostToDevice);

	update_weights<<<NEURONS_QUANTITY*LAYERS_QUANTITY, THREADS_PER_BLOCK>>>(d_input, d_deltas, d_weights, d_biases, d_last_outputs);

	// bring result from device to host
	// todo: maybe is unnecesary if everything is parallel
	//copy_weights_to_host();
	//copy_biases_to_host();
	cudaDeviceSynchronize();

	// free auxiliary data
	cudaFree(d_input);
	return;
}
__host__ int sequential_update_weights(double* input) {
	int input_size = INPUT_SIZE;
	for (int layer_index = 0; layer_index < LAYERS_QUANTITY; layer_index++) {
		if (layer_index > 0) {
			// Si no es input layer, los input vienen de layers previas
			input = &last_outputs[layer_neuron_map(layer_index-1, 0)];// direccion de primer elemento de la layer
			input_size = NEURONS_QUANTITY;
		}
		for (int neuron_index = 0; neuron_index < NEURONS_QUANTITY;
				neuron_index++) {
			// se actualiza su peso
			double neuron_delta = deltas[layer_neuron_map(layer_index,
					neuron_index)];
			if (layer_index != 0 && input_size != NEURONS_QUANTITY) {
				printf("updateWeights :: input and weight size incoherence.\n");
				return -1;
			}
			double real_delta = (LEARNING_RATE * neuron_delta);

			for (int input_index = 0; input_index < input_size; input_index++) {
				weights[layer_neuron_map(layer_index, neuron_index)][input_index] += real_delta*input[input_index];
			}
			biases[layer_neuron_map(layer_index, neuron_index)] += real_delta;
		}
	}
	return 0;
}


/**************************************************************/
/** UTILS **//** UTILS **//** UTILS **//** UTILS **/
/**************************************************************/	
__host__ __device__ int layer_neuron_map(int layer_index, int neuron_index) {
	return neuron_index + layer_index * NEURONS_QUANTITY;
}

__host__ __device__ double sigmoid(double input) {

	return 1.0 / (1.0 + exp(-input));
}
__host__ __device__ double transfer_derivative(double output) {
	return output * (1.0 - output);
}

__device__ double atomicDoubleAdd(double* address, double val){
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
	old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

void intHandler(int dummy) {

	print_stats();
	printf("HASTA PRONTO!\n");
	exit(1);
}

void print_stats(){
	float neurons_feeded_per_sec = neurons_feeded/seconds_feeding;
	float neurons_backp_per_sec = neurons_back_propagated/seconds_back_propagating;
	float neurons_updated_per_sec = neurons_updated_weights/seconds_updating_weights;
	float neurons_evaluated_per_sec = (neurons_feeded_per_sec + neurons_backp_per_sec + neurons_updated_per_sec)/3;
	if (gpu_mode){
			printf("\n///////////////// ESTADISTICAS DE ENTRENAMIENTO CON GPU /////////////////////\n");

		} else {
						printf("\n///////////////// ESTADISTICAS DE ENTRENAMIENTO CON CPU /////////////////////\n");

			}
	printf("- Neuronas evaluadas: %d neuronas\n", neurons_feeded + neurons_back_propagated + neurons_updated_weights);
	printf("- Segundos evaluando neuronas: %f segundos\n", seconds_feeding + seconds_back_propagating + seconds_updating_weights);
	printf("- Neuronas totales evaluadas por segundo: %.f celulas/segundo\n", neurons_evaluated_per_sec);
	printf("- Neuronas 'feeded' por segundo: %.f celulas/segundo\n", neurons_feeded_per_sec);
	printf("- Neuronas 'back propagated' por segundo: %.f celulas/segundo\n", neurons_backp_per_sec);
	printf("- Neuronas actualizadas por segundo: %.f celulas/segundo\n", neurons_updated_per_sec);

	printf("- Cantidad de nueronas: %d neruonas (%d  capas x %d n/c)\n", NEURONS_QUANTITY*LAYERS_QUANTITY, LAYERS_QUANTITY, NEURONS_QUANTITY);
	 seconds_feeding = 0;
	 seconds_updating_weights = 0;
	 seconds_back_propagating = 0;
	  neurons_feeded = 0;
	  neurons_back_propagated = 0;
	  neurons_updated_weights = 0;	
	}

/**************************************************************/
/** MEMORY MANAGEMENT **/ /** MEMORY MANAGEMENT **/
/** MEMORY MANAGEMENT **/ /** MEMORY MANAGEMENT **/
/**************************************************************/	

/**
 *
 * recibe:
 * input_size: tamano del input que debe soportar la red.
 *
 * Inicializa red neuronal en variables locales:
 * last_output: inicializa en 0, en arreglo de doubles tamano *twod_array_size*.
 * deltas: inicializa en 0, en arreglo de doubles tamano *twod_array_size*.
 * bias: inicializa en valores aleatorios entre 0 y 1, en arreglo de doubles tamano *twod_array_size*.
 * weights: inicializa en valores aleatorios entre 0 y 1, en arreglo de doubles tamano *weights_array_size*.
 *
 * Con variables:.
 * twod_array_size: resultado de mapear una red neuronal de *LAYERS_QUANTITY* capas, con *layers_size_array* cantidad de neuronas en cada una, (2dim) en un arreglo 1-dimensional.
 *
 */
__host__ int init_network_struct(int input_size, bool verbose) {
	srand(time(NULL));   // should only be called once

	int twod_array_size = 0;
	twod_array_size = LAYERS_QUANTITY * NEURONS_QUANTITY;

	biases = (double*) malloc(sizeof(double) * twod_array_size);
	last_outputs = (double*) malloc(sizeof(double) * twod_array_size);
	deltas = (double*) malloc(sizeof(double) * twod_array_size);
	weights = (double**) malloc(sizeof(double*) * twod_array_size);

	if (biases == NULL || last_outputs == NULL || deltas == NULL
			|| weights == NULL) {
		printf("Error allocating memory!\n"); //print an error message
		return 1; //return with failure
	}
	// crear areglo de last_output y delta en 0 y bias con valores aleatorios entre 0 y 1

	// iniciar biases aletorios
	for (int i = 0; i < twod_array_size; i++) {
		biases[i] = (rand() % 100 + 1) * 1.0 / 100;
		if (verbose)
			printf("init:: added bias %f to index %d\n", biases[i], i);
	}

	// iniciar weights aletorios
	//caso base
	for (int neuron_index = 0; neuron_index < NEURONS_QUANTITY;
			neuron_index++) {

		// iterar por inputs que puede recibir, para todas neuronas misma cantidad
		weights[layer_neuron_map(0, neuron_index)] = (double*) malloc(
				sizeof(double) * INPUT_SIZE);

		for (int input_index = 0; input_index < input_size; input_index++) {
			weights[layer_neuron_map(0, neuron_index)][input_index] = (rand()% 100 + 1) * 1.0 / 100;
			if (verbose)
				printf(
						"init:: added weight %f to input %d of neuron %d of first layer\n",
						weights[layer_neuron_map(0, neuron_index)][input_index],
						input_index, neuron_index);
		}

	}
	// caso iterativo
	for (int layer_index = 1; layer_index < LAYERS_QUANTITY; layer_index++) {
		// iterar por capas
		for (int neuron_index = 0; neuron_index < NEURONS_QUANTITY;
				neuron_index++) {
			// iterar por neuronas
			weights[layer_neuron_map(layer_index, neuron_index)] =
					(double*) malloc(sizeof(double) * LAYERS_QUANTITY);

			if (weights[layer_neuron_map(0, neuron_index)] == NULL) {
				printf("Error allocating memory!\n"); //print an error message
				return 1; //return with failure
			}
			for (int input_index = 0; input_index < NEURONS_QUANTITY;
					input_index++) {
				weights[layer_neuron_map(layer_index, neuron_index)][input_index] =
						(rand() % 100 + 1) * 1.0 / 100;
				if (verbose)
					printf(
							"init:: added weight %f to input %d of neuron %d of layer %d\n",
							weights[layer_neuron_map(layer_index, neuron_index)][input_index],
							input_index, neuron_index, layer_index);
			}
		}

	}
	cudaMalloc((void **) &d_last_outputs, sizeof(double) * twod_array_size);
	cudaMalloc((void **) &d_deltas, sizeof(double) * twod_array_size);
	cudaMalloc((void **) &d_biases, sizeof(double) * twod_array_size);

	alloc_weights_in_gpu();
	copy_weights_to_gpu(false);
	copy_biases_to_gpu();

	cudaError_t err = cudaSuccess;
	cudaMemcpy(d_last_outputs, last_outputs, sizeof(double)*twod_array_size,
			cudaMemcpyHostToDevice);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error d_last_outputs: %s\n", cudaGetErrorString(err));
	}

	cudaMemcpy(d_deltas, deltas, sizeof(double)*twod_array_size, cudaMemcpyHostToDevice);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error d_deltas: %s\n", cudaGetErrorString(err));
	}

	return 0;

}

int destroy_net() {
	cudaFree(d_biases);
	cudaFree(d_biases);
	cudaFree(d_last_outputs);
	free(biases);
	free(deltas);
	free(last_outputs);
	free_weights_in_gpu();

	for (int i = 0; i < LAYERS_QUANTITY * NEURONS_QUANTITY; i++) {
		free(weights[i]);
	}
	free(weights);
	return 0;
}

__host__ int free_weights_in_gpu(){
	cudaFree(d_weights);
	for (int i = 0; i < NEURONS_QUANTITY*LAYERS_QUANTITY; i++){
		cudaFree(temp_d_ptrs[i]);
	}
	return 1;
}

__host__ int alloc_weights_in_gpu(){
	int twod_array_size = LAYERS_QUANTITY*NEURONS_QUANTITY;
	cudaError_t err = cudaSuccess;

	cudaMalloc((void **) &d_weights, sizeof(double*) * twod_array_size);
	temp_d_ptrs = (void**) malloc(sizeof(void*) * twod_array_size);
	int layer_index = 0;
		for (int neuron_index = 0; neuron_index < NEURONS_QUANTITY; neuron_index++){
			cudaMalloc((void**) &temp_d_ptrs[layer_neuron_map(layer_index, neuron_index)], sizeof(double) * INPUT_SIZE);
			if (err != cudaSuccess) {
				printf("error cudaMalloctemp_d_ptrs: %s\n", cudaGetErrorString(err));
			}
		}
		for (;layer_index < LAYERS_QUANTITY; layer_index++){
			for (int neuron_index = 0; neuron_index < NEURONS_QUANTITY; neuron_index++){
				cudaMalloc((void**) &temp_d_ptrs[layer_neuron_map(layer_index, neuron_index)], sizeof(double) * NEURONS_QUANTITY);
				if (err != cudaSuccess) {
					printf("error cudaMalloctemp_d_ptrs: %s\n", cudaGetErrorString(err));
				}
			}

		}
		return 1;
}

__host__ int copy_weights_to_gpu(bool is_update){
	int twod_array_size = LAYERS_QUANTITY*NEURONS_QUANTITY;
	if (is_update){
		free_weights_in_gpu();
		alloc_weights_in_gpu();
	}

	cudaError_t err = cudaSuccess;
	int layer_index = 0;
	for (int neuron_index = 0; neuron_index < NEURONS_QUANTITY; neuron_index++){
		cudaMalloc((void**) &temp_d_ptrs[layer_neuron_map(layer_index, neuron_index)], sizeof(double) * INPUT_SIZE);
		if (err != cudaSuccess) {
			printf("error cudaMalloctemp_d_ptrs: %s\n", cudaGetErrorString(err));
		}
		cudaMemcpy(temp_d_ptrs[layer_neuron_map(layer_index, neuron_index)],
				weights[layer_neuron_map(layer_index, neuron_index)],
				sizeof(double) * INPUT_SIZE,
				cudaMemcpyHostToDevice);
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("error cudaMemcpytemp_d_ptrs: %s\n", cudaGetErrorString(err));
		}
	}
	for (;layer_index < LAYERS_QUANTITY; layer_index++){
		for (int neuron_index = 0; neuron_index < NEURONS_QUANTITY; neuron_index++){
			cudaMalloc((void**) &temp_d_ptrs[layer_neuron_map(layer_index, neuron_index)], sizeof(double) * NEURONS_QUANTITY);
			if (err != cudaSuccess) {
				printf("error cudaMalloctemp_d_ptrs: %s\n", cudaGetErrorString(err));
			}
			cudaMemcpy(temp_d_ptrs[layer_neuron_map(layer_index, neuron_index)],
					weights[layer_neuron_map(layer_index, neuron_index)],
					sizeof(double) * NEURONS_QUANTITY,
					cudaMemcpyHostToDevice);
			err = cudaGetLastError();
			if (err != cudaSuccess) {
				printf("error cudaMemcpytemp_d_ptrs: %s\n", cudaGetErrorString(err));
			}
		}

	}
	cudaMemcpy(d_weights, temp_d_ptrs, sizeof(double*) * twod_array_size,
			cudaMemcpyHostToDevice);
	return 1;
}

__host__ int copy_weights_to_host(){
	cudaError_t err = cudaSuccess;

	int layer_index = 0;
	for (int neuron_index = 0; neuron_index < NEURONS_QUANTITY; neuron_index++){
		cudaMemcpy(weights[layer_neuron_map(layer_index, neuron_index)],
		temp_d_ptrs[layer_neuron_map(layer_index, neuron_index)],
		sizeof(double) * INPUT_SIZE,
		cudaMemcpyDeviceToHost);
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("error cudaMemcpytemp_d_ptrs: %s\n", cudaGetErrorString(err));
		}



	}

	for (;layer_index < LAYERS_QUANTITY; layer_index++){
		for (int neuron_index = 0; neuron_index < NEURONS_QUANTITY; neuron_index++){

			cudaMemcpy(weights[layer_neuron_map(layer_index, neuron_index)],
			temp_d_ptrs[layer_neuron_map(layer_index, neuron_index)],
			sizeof(double) * NEURONS_QUANTITY,
			cudaMemcpyDeviceToHost);
			err = cudaGetLastError();
			if (err != cudaSuccess) {
				printf("error cudaMemcpytemp_d_ptrs: %s\n", cudaGetErrorString(err));
			}
		}
	}
	return 1;
}

__host__ int copy_biases_to_gpu(){
	int twod_array_size = LAYERS_QUANTITY*NEURONS_QUANTITY;
	cudaError_t err = cudaSuccess;

	cudaMemcpy(d_biases, biases, sizeof(double)*twod_array_size, cudaMemcpyHostToDevice);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error d_biases: %s\n", cudaGetErrorString(err));
	}
	return 1;
}

__host__ int copy_biases_to_host(){
	int twod_array_size = LAYERS_QUANTITY*NEURONS_QUANTITY;
	cudaError_t err = cudaSuccess;

	cudaMemcpy(biases, d_biases, sizeof(double)*twod_array_size, cudaMemcpyDeviceToHost);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error d_biases: %s\n", cudaGetErrorString(err));
	}
	return 1;
}

