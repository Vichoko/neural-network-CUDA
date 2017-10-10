#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

// read file
#include <stdio.h>
#include <stdlib.h>

// strcmp
#include <string.h>

// vecto#include <vector>
#include <vector>
#include <iostream>

// hash map
#include <map>

#define BUFFER_SIZE 900

typedef struct {
	double** ham_bag;
	double** spam_bag;
	int dict_size;
	int ham_data_len;
	int spam_data_len;
} data_container_t;


typedef struct {
	char** spam_msgs;
	char** ham_msgs;
	int ham_data_len;
	int spam_data_len;
} raw_data_container_t;



raw_data_container_t*  parse(const char* filename, bool verbose);
data_container_t* preprocess_texts(raw_data_container_t* raw_data);

