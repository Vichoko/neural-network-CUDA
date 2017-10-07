#include "neural-network.h"



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

#define THREADS_PER_BLOCK 8 // tiene que ser potencia de 2 y mayor o igual que la cantidad maxima de neuronas que puede haber por capa
#define LEN(x)  (sizeof(x) / sizeof((x)[0]))

#define INPUT_SIZE 2
#define NEURONS_QUANTITY 2
#define LAYERS_QUANTITY 2

// GLOBAL VARS
int layers_count = LAYERS_QUANTITY;
double LEARNING_RATE = 0.1;

// utils
/**
 * Mapeo layer, neuuroca con capas con distintas neuronas.
 __host__ __device__ int layer_neuron_map(int layer_index, int neuron_index,
 int* layers_sizes_array) {
 int offset = 0;
 for (int i = 0; i < layer_index; i++) {
 offset += layers_sizes_array[i];
 }
 return neuron_index + offset;
 }
 * */
/**
 * Mapeo layer, neurona con capas con misma neuronas.
 *
 * */
__host__ __device__ int layer_neuron_map(int layer_index, int neuron_index) {
	return neuron_index + layer_index * NEURONS_QUANTITY;
}

__device__ double sigmoid(double input) {

	return 1.0 / (1.0 + exp(-input));
}
double transfer_derivative(double output) {
	return output * (1.0 - output);
}
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
			if (err != cudaSuccess) {
				printf("error cudaMemcpytemp_d_ptrs: %s\n", cudaGetErrorString(err));
			}
		}

	}
	cudaMemcpy(d_weights, temp_d_ptrs, sizeof(double*) * twod_array_size,
			cudaMemcpyHostToDevice);
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

__host__ int init_network_struct(int input_size, bool verbose) {
	srand(time(NULL));   // should only be called once

	/*
	 neuron_counts_array = (int*) malloc(sizeof(int) * LAYERS_QUANTITY);
	 if (neuron_counts_array == NULL) {
	 printf("Error allocating memory!\n"); //print an error message
	 return 1; //return with failure
	 }
	 //for (int layer_index = 0; layer_index < LAYERS_QUANTITY; layer_index++) {
	 //neuron_counts_array[layer_index] = NEURONS_QUANTITY;
	 //if (verbose)
	 //printf("init:: added %d neurons to layer %d\n",
	 //neuron_counts_array[layer_index], layer_index);
	 //}
	 *
	 * */

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
		biases[i] = 0.1;

		//biases[i] = (rand() % 100 + 1) * 1.0 / 100;
		if (verbose)
			printf("init:: added bias %f to index %d\n", biases[i], i);
	}

	// iniciar weights aletorios
	//caso base
	//for (int neuron_index = 0; neuron_index < neuron_counts_array[0]; neuron_index++) {
	for (int neuron_index = 0; neuron_index < NEURONS_QUANTITY;
			neuron_index++) {

		// iterar por inputs que puede recibir, para todas neuronas misma cantidad
		weights[layer_neuron_map(0, neuron_index)] = (double*) malloc(
				sizeof(double) * input_size);

		for (int input_index = 0; input_index < input_size; input_index++) {
			weights[layer_neuron_map(0, neuron_index)][input_index] = 0.5;
//			weights[layer_neuron_map(0, neuron_index)][input_index] = (rand()
//					% 100 + 1) * 1.0 / 100;
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
						0.5;
//				weights[layer_neuron_map(layer_index, neuron_index)][input_index] =
//						(rand() % 100 + 1) * 1.0 / 100;
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

/*int main(void) {


 init_network_struct(INPUT_SIZE);

 int a, b, c; // host copies of a, b, c
 int *d_a, *d_b, *d_c; // device copies of a, b, c
 int size = sizeof(int);
 // Allocate space for device copies of a, b, c
 cudaMalloc((void **) &d_a, size);
 cudaMalloc((void **) &d_b, size);
 cudaMalloc((void **) &d_c, size);
 // Setup input values
 a = 2;
 b = 7;
 // Copy inputs to device
 cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
 cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
 // Launch add() kernel on GPU
 // Copy result back to host
 cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
 printf("hola: %d", c);
 // Cleanup
 cudaFree(d_a);
 cudaFree(d_b);
 cudaFree(d_c);
 return 0;
 }*/
__device__ double atomicDoubleAdd(double* address, double val)
{
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

__global__ void first_layer_synapsis(double* input,
		double** weights, double* biases, double* last_outputs) {
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
//			printf("procesando input: %d de neurona: %d de layer: %d, con weigth: %f, bias: %f\n", input_index, neuron_index, 0, weights[layer_neuron_map(0, neuron_index)][input_index], biases[layer_neuron_map(0, neuron_index)]);
//			printf("input: %f de entrada; acumulador queda: %f\n", input[input_index], acc);


			__syncthreads(); // esperar a que se calcule suma ponderada todos los thread

			if (threadIdx.x == 0){
				double result = acc
						+ biases[layer_neuron_map(0, neuron_index)];
				//printf("calculated result: %f with sigmoid; %f (bias: %f); in layer: %d, in neuron: %d\n", result, sigmoid(result), biases[layer_neuron_map(0, neuron_index)], 0, neuron_index);
				last_outputs[layer_neuron_map(0, neuron_index)] = sigmoid(result);
			}
		}
	}
}
__global__ void non_first_layer_synapsis(int* layer_index, double** weights,
		double* biases, double* last_outputs) {
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
//			printf("procesando input: n° %d de neurona: %d de layer: %d, con weigth: %f, bias: %f\n", input_index, neuron_index, *layer_index, weights[layer_neuron_map(*layer_index, neuron_index)][input_index], biases[layer_neuron_map(*layer_index, neuron_index)]);
//			printf("input: %f de neurona: %d,%d; acumulador queda: %f\n\n", last_outputs[layer_neuron_map((*layer_index) - 1, input_index)], input_index, (*layer_index) - 1, acc);

			//printf("input: %f, weigth: %f, bias: %f, accomulated result: %f\n", last_outputs[layer_neuron_map((*layer_index) - 1, input_index)], weights[layer_neuron_map(*layer_index, neuron_index)][input_index], biases[layer_neuron_map(*layer_index, neuron_index)], acc);

			__syncthreads(); // esperar a que se calcule suma ponderada todos los thread

			if (threadIdx.x == 0){
				double result = acc
						+ biases[layer_neuron_map(*layer_index, neuron_index)];
				//printf("calculated result: %f with sigmoid; %f (bias: %f); in layer: %d, in neuron: %d\n", result, sigmoid(result), biases[layer_neuron_map(*layer_index, neuron_index)], *layer_index, neuron_index);
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

	cudaMemcpy(d_input, input, sizeof(double) * INPUT_SIZE,
			cudaMemcpyHostToDevice);
	copy_weights_to_gpu(true);
	copy_biases_to_gpu();

// caso base
	first_layer_synapsis<<<
			NEURONS_QUANTITY,
			THREADS_PER_BLOCK>>>(d_input, d_weights, d_biases,
			d_last_outputs);

	cudaDeviceSynchronize(); // esperar que termine primera capa

// caso iterativo
	for (int layer_index = 1; layer_index < LAYERS_QUANTITY; layer_index++) {
		cudaMemcpy(d_layer_index, &layer_index, sizeof(int),
				cudaMemcpyHostToDevice);
		// actualizar layer_index

		non_first_layer_synapsis<<<
				NEURONS_QUANTITY,
				THREADS_PER_BLOCK>>>(d_layer_index, d_weights, d_biases,
				d_last_outputs);
		cudaDeviceSynchronize(); // esperar que termine segunda capa
	}
// bring result
	cudaMemcpy(last_outputs, d_last_outputs,
			sizeof(double) * NEURONS_QUANTITY * LAYERS_QUANTITY,
			cudaMemcpyDeviceToHost);

	double* output = (double*) malloc(sizeof(double) * NEURONS_QUANTITY);
	memcpy(output, &(last_outputs[layer_neuron_map(LAYERS_QUANTITY - 1, 0)]),
			sizeof(double) * NEURONS_QUANTITY);

	cudaFree(d_input);
	cudaFree(d_layer_index);
	return;
}

__host__ int sequential_back_propagation(double* expected_output, bool verbose) {
	for (int layer_index = LAYERS_QUANTITY - 1; layer_index >= 0;
			layer_index--) {
		if (verbose)
			printf("back propagating layer: %d\n", layer_index);
		// backward iteration. No paralelizable (se necesita resultado previo para siguiente capa).
		if (layer_index == LAYERS_QUANTITY - 1) {
			// Caso capa de salida
			//if (verbose)
				//printf("base case, output layer: %d\n", layer_index);
			for (int neuron_index = 0; neuron_index < NEURONS_QUANTITY;
					neuron_index++) {
				// todo: paralelizar este for
				// ultimo output generado por esta neurona
				double actual_output = last_outputs[layer_neuron_map(
						layer_index, neuron_index)];
				if (verbose) {
					printf("	expected output: %f\n", expected_output[neuron_index]);
					printf("	actual output got: %f\n",
							actual_output);
				}

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
						printf("		neighbor layer weigth are null.\n");
						return -1;
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
				if (verbose)
					printf("	delta of neuron updated to : %f\n",
							deltas[layer_neuron_map(layer_index, neuron_index)]);

			}

		}

	}
	return 0;

}

__host__ int sequential_update_weights(double* input) {
	int input_size = INPUT_SIZE;
	for (int layer_index = 0; layer_index < LAYERS_QUANTITY; layer_index++) {
		if (layer_index > 0) {
			// Si no es input layer, los input vienen de layers previas
			input = &last_outputs[layer_neuron_map(layer_index, 0)];// direccion de primer elemento de la layer
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

__host__ int train(double** input, double** expected_output, int data_count,
		int n_epochs, bool run_in_parallel, bool verbose) {

	int output_size = NEURONS_QUANTITY; //todo: variable

// recibe dataset de entrenamiento; varios input con sus respectivos output
	if (sizeof(input) != sizeof(expected_output)) {
		printf(
				"train :: dataset input and expectedOutput arrays have different lenghts: %lu and %lu.\n",
				sizeof(input), sizeof(expected_output));
		return -1;
	}
	printf("learnRate: %f\n", LEARNING_RATE);
	printf("numer of epochs: %d\n", n_epochs);
//double errors[nEpochs];
// Siguientes variables son para no saturar de impresiones en consola.
	int epochsPart = n_epochs / 1000;
	int counter = 0;
	if (verbose) {
		printf("initializing epochs iteration.\n");
	}


	for (int epoch_index = 0; epoch_index < n_epochs; epoch_index++) {
		// epochs se ejecutan en secuencia. No paralelizable (se necesita resultado previo para el siguiente).
		double sumError = 0;

		if (verbose) {
			printf("epoch %d.\n", epoch_index);
		}

		for (int data_index = 0; data_index < data_count; data_index++) {
			// entrenar sobre cada par de vectores input/output. No paralelizable (se necesita exclusión mutua durante el proeceso).
			if (verbose) {
				printf("forward feeding with data index: %d.\n", data_index);
			}


			if (run_in_parallel){
				parallel_forward_feed(input[data_index]);
			} else {
				parallel_forward_feed(input[data_index]); // todo: implement sequential
			}
			double* actual_output = &last_outputs[layer_neuron_map(LAYERS_QUANTITY - 1, 0)];
			if (verbose) { //todo: borrar
				printf("	done forward feeding.\n");
			}

			if (sizeof(actual_output) != sizeof(expected_output[data_index])) {
				printf(
						"train :: one of layers actualOutput/expectedOutput have different sizes.\n");
				return -2;
			}

			for (int outputIndex = 0; outputIndex < output_size;
					outputIndex++) {
				// Para cada input se calcula el error cuadratico medio para visualizar aprendizaje
				sumError += (expected_output[data_index][outputIndex]
						- actual_output[outputIndex])
						* (expected_output[data_index][outputIndex]
								- actual_output[outputIndex]);

			}
			//free(actual_output); // not more use needed

			if (verbose) {
				printf("done calculating error\n");
			}
			if (verbose) {
				printf("back propagating with data index: %d.\n", data_index);
			}

			if (run_in_parallel){
				sequential_back_propagation(expected_output[data_index], false); // todo: implement parallel
			} else {
				sequential_back_propagation(expected_output[data_index], false);
			}


			if (verbose) {
				printf("	done back propagating.\n");
			}
			if (verbose) {
				printf("updating weigths with data index: %d.\n", data_index);
			}

			if (run_in_parallel){
				sequential_update_weights(input[data_index]); // todo: implement parallel
			} else {
				sequential_update_weights(input[data_index]);
			}

			if (verbose) {
				printf("	done updating weigths.\n");
			}
		}
		if (++counter >= epochsPart) {
			printf("Epoch: %d , error: %f\n", epoch_index, sumError);
			counter = 0;
		}
		//errors[epochIndex] = sumError;
		// debug

		/*		if (epochIndex > 3 && 0) {
		 if (errors[epochIndex - 1] == errors[epochIndex - 2] && errors[epochIndex - 1] == errors[epochIndex]) {
		 // si error no cambia en 3 ultimas epocas, probablemente no cambie mas.
		 printf("train :: finishing train because of no-change in error.");
		 break;
		 }
		 }*/
	}
	return 0;
}

// TESTS
__host__ int networkLogicGateSingleOutputTest(double** input,
		double** expected_output, int data_count, bool verbose) {
	if (verbose) {
		printf("initializing net.\n");
	}
	init_network_struct(INPUT_SIZE, true);
	if (verbose) {
		printf("	done initializing net.\n");
	}
	/** Red con 2 neuronas de entrada, 2 escondidas y una de salida, generada con pesos aleatorios, para aprender XOR.
	 * Clases binarias:
	 * 	1, si bit_1 <GATE> bit_2 == 1;
	 * 	0 si no.*/

	if (verbose) {
		printf("initializing training.\n");
	}
	bool run_in_parallel = true;
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

int main(void) {
	networkANDSingleOutputLearningTest(false);
}
