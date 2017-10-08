#include "tests.h"

std::vector<char*> spam_msgs; 
std::vector<char*> ham_msgs;
std::map<std::string, char*> dict;

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

double** preprocess_texts(std::vector<char*> msgs){
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
	// tengo arreglo de string {"hola que tal","vente pa aki", ...}
	// tengo que tener arreglo de arreglo de strings { {hola, que, tal}, {vente, pa, aki}, etc}
	
	// remove stopwords
	const char* stop_words[] = {"a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"};
	
	// hacer stemming
	// calcular td-idf
	// retornar datos procesados
	return NULL;
}
