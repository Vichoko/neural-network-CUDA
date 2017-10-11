/**
 * Autor: Vicente Oyanedel M.
 * Fecha: Otubre, 2017.
 * 
 * */
#include "parser.h"
#include "stemmer.cpp"

const char* stop_words[] = {"a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"};


float train(double** input, double** expected_output, int data_count, 
	int n_epochs, bool run_in_parallel, bool verbose);



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


raw_data_container_t* parse(const char* filename, bool verbose){
	std::vector<char*> spam_msgs; 
	std::vector<char*> ham_msgs;
	
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
    char** spam_msgs_array = (char**) malloc(sizeof(char*) * spam_msgs.size());
    char** ham_msgs_array = (char**) malloc(sizeof(char*) * ham_msgs.size());
    for (int i = 0; i < spam_msgs.size(); i++){
    	spam_msgs_array[i] = spam_msgs.at(i);
    }
    for (int i = 0; i < ham_msgs.size(); i++){
    	ham_msgs_array[i] = ham_msgs.at(i);
    }

    raw_data_container_t* ret = (raw_data_container_t*) malloc(sizeof(raw_data_container_t));
    ret->ham_msgs = ham_msgs_array;
    ret->spam_msgs = spam_msgs_array;
    ret->ham_data_len = ham_msgs.size();
    ret->spam_data_len = spam_msgs.size();
    return ret;
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

data_container_t* preprocess_texts(raw_data_container_t* raw_data){
    struct stemmer * z = create_stemmer();

	printf("borrando simblos especiales y dejando en minusculas\n");
	// 
	// eliminar simbolos especiales y dejar en minusculas HAM
	for(int ham_index = 0; ham_index < raw_data->ham_data_len; ham_index++){
		char* real_string = raw_data->ham_msgs[ham_index];
		// iterar sobre las palabras
		char aux_string[max_str_len];
		int aux_index = 0;
		for (int i = 0; i < strlen(real_string); i++){
			// iterar sobre los caracteres
			if (!is_special(real_string[i])){
				aux_string[aux_index] = tolower(real_string[i]);
				aux_index++;
			}
		}
		aux_string[aux_index] = '\0';
		memcpy(real_string, aux_string, sizeof(char)*strlen(aux_string));
	}
	// 
	// eliminar simbolos especiales y dejar en minusculas SPAM
	for(int spam_index = 0; spam_index < raw_data->spam_data_len; spam_index++){
		char* real_string = raw_data->spam_msgs[spam_index];
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
	std::vector<std::vector<int> > tokenized_ham_msgs;
	std::vector<std::vector<int> > tokenized_spam_msgs;

	for(int ham_index = 0; ham_index < raw_data->ham_data_len; ham_index++){
		char* real_string = raw_data->ham_msgs[ham_index]; // mensaje completo (frase = strings separados por espacio)
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
		tokenized_ham_msgs.push_back(phrase);
	}

	for(int spam_index = 0; spam_index < raw_data->spam_data_len; spam_index++){
		char* real_string = raw_data->spam_msgs[spam_index]; // mensaje completo (frase = strings separados por espacio)
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
		tokenized_spam_msgs.push_back(phrase);
	}
	// tengo que tener arreglo de arreglo de strings { {1, 2, 3}, {4, 3, 2}, etc}
	printf("	done\n");
	printf("size of dict is %d\n", dict_size);


	// calcular td-idf
	printf("Comenzando calculo de bag of words \n");
	// matriz de <cantidad de palabras> caracteristicas
	double** ham_bag_of_words = (double**) malloc(sizeof(double*) * tokenized_ham_msgs.size());
	double** spam_bag_of_words = (double**) malloc(sizeof(double*) * tokenized_spam_msgs.size());
	for (int i = 0; i < tokenized_ham_msgs.size(); i++){
		ham_bag_of_words[i] = (double*) malloc(sizeof(double) * dict_size);
	}
	for (int i = 0; i < tokenized_spam_msgs.size(); i++){
		spam_bag_of_words[i] = (double*) malloc(sizeof(double) * dict_size);
	}
	
	int phrase_index = 0;
	for(std::vector<std::vector<int> >::iterator it = tokenized_ham_msgs.begin(); it != tokenized_ham_msgs.end(); it++){	// retornar datos procesados+
		std::vector<int> phrase = *it;

		for(std::vector<int>::iterator it2 = phrase.begin(); it2 != phrase.end(); it2++){
			int current_word = *it2;
			ham_bag_of_words[phrase_index][current_word] += 1; // frecuencia
		}
		phrase_index++;
	}

	phrase_index = 0;
	for(std::vector<std::vector<int> >::iterator it = tokenized_spam_msgs.begin(); it != tokenized_spam_msgs.end(); it++){	// retornar datos procesados+
		std::vector<int> phrase = *it;

		for(std::vector<int>::iterator it2 = phrase.begin(); it2 != phrase.end(); it2++){
			int current_word = *it2;
			spam_bag_of_words[phrase_index][current_word] += 1; // frecuencia
		}
		phrase_index++;
	}
	printf("	done");

	free_stemmer(z);
	data_container_t* ret = (data_container_t*) malloc(sizeof(data_container_t));
	ret->ham_bag = ham_bag_of_words;
	ret->spam_bag  = spam_bag_of_words;
	ret->dict_size = dict_size;
	ret->ham_data_len = tokenized_ham_msgs.size();
	ret->spam_data_len = tokenized_spam_msgs.size();
	return ret;
}

