#include "tests.h"
#include "stemmer.cpp"

std::vector<char*> spam_msgs; 
std::vector<char*> ham_msgs;
std::map<std::string, unsigned int> dict;
int dict_size = 0;

int max_str_len = 0; 

void refresh_max_str_len(int len){
	if (max_str_len < len){
		max_str_len = len;
	}
}

const char* get_csv_field(char* line, int num)
{
	const char* tok;
	for (tok = strtok(line, ",");
			tok && *tok;
			tok = strtok(NULL, ",\n"))
	{
		if (!--num)
			return tok;
	}
	return NULL;
}


 
int main()
{
	parse("spam.csv", true);
	preprocess_texts(ham_msgs);
	preprocess_texts(spam_msgs);
}

int parse(const char* filename, bool verbose){
	FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    fp = fopen(filename, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);
    ///////////// FILL BUFFER OF MESSAGES 
	int first_line = 1;
    while ((read = getline(&line, &len, fp)) != -1) {
    	    if (first_line){
    	    	first_line = 0;
    	    	continue;
    	    } 

    	    const char* msg = get_csv_field(line, 2);
    	    const char* clas = get_csv_field(line, 1);

    	    int msg_len = strlen(msg);
    	    refresh_max_str_len(msg_len);
    	    char* aux_ptr = (char*) malloc(sizeof(char) * msg_len);
    	    memcpy(aux_ptr, msg, sizeof(char) * msg_len);

    	if (strcmp(clas, "ham") == 0){
    		ham_msgs.push_back(aux_ptr);
    	} else {
    		spam_msgs.push_back(aux_ptr);
    	}
    }

    fclose(fp);
    if (line)
        free(line);

    if (verbose){
    	printf("%lu Ham, %lu spam \n", ham_msgs.size(), spam_msgs.size());
    }
}

/***
* Is special if isnt a letter, number or space.
*/
int is_special(char c){
	if (!('a' <= c && 'z' >= c) &&
	     !('A' <= c && 'Z' >= c) &&
	     !('0' <= c && '9' >= c) && 
	     c != ' '){
		return 1;
	}

	if (c == ',' || c == '.' || c == '\r'){
		return 1;
	}
	return 0;
}
int add_word_to_dict(char* word){
	std::string str (word);
	std::map<std::string, unsigned int>::iterator it = dict.find(str);
	int ptr;
	if(it != dict.end())
	{
	   //element found;
	   ptr = it->second;
	   return ptr;
	}

	// need to add word to dict
	dict.insert(std::pair<std::string, int>(str, dict_size));
	return dict_size++;
}
int is_stop_word(char* word){
	for (int i = 0; i < sizeof(stop_words)/sizeof(char*); i++){
		if (strcmp(word, stop_words[i]) == 0){
			//printf("%s is s-w\n", word);
			return 1;
		}
	}
	return 0;
}

double** preprocess_texts(std::vector<char*> msgs){
    struct stemmer * z = create_stemmer();

	// eliminar simbolos especiales y dejar en minusculas
	printf("borrando simblos especiales y dejando en minusculas\n");
	for(std::vector<char*>::iterator it = msgs.begin(); it != msgs.end(); it++){
		char* real_string = *it;
		// iterar sobre las palabras
		char aux_string[max_str_len];
		int aux_index = 0;
		for (int i = 0; i < strlen(real_string); i++){
			if (!is_special(real_string[i])){
				aux_string[aux_index] = tolower(real_string[i]);
				aux_index++;
			}
		}
		aux_string[aux_index] = '\0';
		memcpy(real_string, aux_string, sizeof(char)*strlen(aux_string));
	}
	printf("	done\n");
	// tengo arreglo de string {"hola que tal","vente tal que", ...}
	// tengo que tener arreglo de arreglo de strings { {1, 2, 3}, {4, 3, 2}, etc}
	printf("eliminando stop words, haciendo stemming, llenando diccionario y transformando frases.\n");
	std::vector<std::vector<int> > tokenized_msgs;
	for(std::vector<char*>::iterator it = msgs.begin(); it != msgs.end(); it++){
		char* real_string = *it; // mensaje completo (frase = strings separados por espacio)
		std::vector<int> phrase;

		char *p = strtok(real_string, " ");
		while(p != NULL) {
			// tokenizer, p es cada palabra de la frase

			if (!is_stop_word(p)){
				// si no es stopword, se agrega a diccionario y se toma en cuenta
				int res = stem(z, p, strlen(p));

				phrase.push_back(add_word_to_dict(p));
			}
		    p = strtok(NULL, " ");
		}
		tokenized_msgs.push_back(phrase);
	}
	// tengo que tener arreglo de arreglo de strings { {1, 2, 3}, {4, 3, 2}, etc}
	printf("	done\n");
	printf("size of dict is %d\n", dict_size);


	// calcular td-idf
	double** bag_of_words = (double**) malloc(sizeof(double) * tokenized_msgs.size());
	for (int i = 0; i < tokenized_msgs.size(); i++){
		bag_of_words[i] = (double*) malloc(sizeof(double) * dict_size);

	}
	//double bag_of_words[tokenized_msgs.size()][dict_size];
	int phrase_index = 0;
	for(std::vector<std::vector<int> >::iterator it = tokenized_msgs.begin(); it != tokenized_msgs.end(); it++){	// retornar datos procesados+
		std::vector<int> phrase = *it;

		for(std::vector<int>::iterator it2 = phrase.begin(); it2 != phrase.end(); it2++){
			int current_word = *it2;
			bag_of_words[phrase_index][current_word] += 1;
		}
		phrase_index++;
	}

	free_stemmer(z);
	return bag_of_words;
}

