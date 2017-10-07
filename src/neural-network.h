/*
 * ex.h
 *
 *  Created on: 24-09-2017
 *      Author: vichoko
 */

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

namespace ex {

class ex {
public:
	ex();
	virtual ~ex();
};

} /* namespace ex */

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// public functions
// constructor
#ifdef __cplusplus
extern "C"
#endif
int init_network_struct(int input_size);
// entrenar
int train(double** input, double** expected_output, int n_epochs);

/** util functions **/
#endif /* NEURAL_NETWORK_H_ */

