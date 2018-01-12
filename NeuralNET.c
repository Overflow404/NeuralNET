#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*double trainSet[4][3] = 	{{ 1.0, 0.0, 0.0 },
							 { 1.0, 0.0, 1.0 }, 	
							 { 1.0, 1.0, 0.0 }, 
							 { 1.0, 1.0, 1.0 }};
							 
double realSet[1][4] = {{ 0.0, 1.0, 1.0, 1.0 }};*/

double trainSet[8][4] = 	{{ 1.0, 0.0, 0.0, 0.0 },
							 { 1.0, 0.0, 0.0, 1.0 }, 	
							 { 1.0, 0.0, 1.0, 0.0 }, 
							 { 1.0, 0.0, 1.0, 1.0 },	
							 { 1.0, 1.0, 0.0, 0.0 }, 	
							 { 1.0, 1.0, 0.0, 1.0 }, 	
							 { 1.0, 1.0, 1.0, 0.0 }, 	
							 { 1.0, 1.0, 1.0, 1.0 }};
							 
double realSet[1][8] = {{ 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0 }};

typedef struct neuron{
	double inputWeight;
	double outputWeight;
}Neuron;

typedef struct layer{
	Neuron** listOfNeurons;
	int numberOfNeurons;
}Layer;

typedef struct brain{
	Layer* inputLayer;
	Layer* outputLayer;
	int startingEpoch;
	int maxEpoch;
	int numbersOfNeuronsInInputLayer;
	int numbersOfNeuronsInOutputLayer;
	int trainSetRows;
	int trainSetColumns;
	double targetError;
	double learningRate;
}Brain;

double initNeuron(){
	double low = -3;
	double high = 3; 
	return (( ( double )rand() * ( high - low ) )
               / ( double )RAND_MAX + low);
}

Layer* initIOL(Layer* layer, int n){
	layer->numberOfNeurons = n;
	Neuron** list = malloc(n*sizeof(Neuron*));
	int i;
	for(i=0;i<layer->numberOfNeurons;i++){
		Neuron* neuron = malloc(sizeof(Neuron));
		neuron->inputWeight = initNeuron();
		neuron->outputWeight = initNeuron();
		list[i] = neuron;
		layer->listOfNeurons = list;
	}
	return layer;
}

double fncStep(double v){
		if (v >= 0)
			return 1.0;
		else
			return 0.0;
}

double tanH(double x){
	return tanh(x);
}

double newWeight(double inputWold, Brain* TB, double error,double trainset){
	return inputWold + TB->learningRate*error*trainset;
}

Neuron** teach(int n, int line, Brain* TB, double net,double error){
	Neuron** list = malloc(n*sizeof(Neuron*));
	double inputWnew;
	double inputWold;
	int j;
	for(j=0;j<n;j++){
		inputWold = TB->inputLayer->listOfNeurons[j]->inputWeight;
		inputWnew = newWeight(inputWold,TB,error,trainSet[line][j]);
		Neuron* n = malloc(sizeof(Neuron));
		n->inputWeight = inputWnew;
		list[j] = n;
	}
	return list;
}

Brain* trainBrain(Brain* TB){
	int i,j;
	while(TB->startingEpoch < TB->maxEpoch){
		double estimatedOutput = 0.0;
		double realOutput = 0.0;
		for(i=0;i<TB->trainSetRows;i++){
			double net = 0.0;
			for(j=0;j<TB->trainSetColumns;j++){
				double N = TB->inputLayer->listOfNeurons[j]->inputWeight;
				net = net + N * trainSet[i][j];
			}
			estimatedOutput = tanH(net);
			realOutput = realSet[0][i];
			double error = realOutput - estimatedOutput;
			if(abs(error) > TB->targetError){
				Layer* inputLayer = malloc(sizeof(Layer));
				inputLayer->listOfNeurons =teach(TB->trainSetRows,i,TB,net,error);
				inputLayer->numberOfNeurons = TB->inputLayer->numberOfNeurons;
				(TB->inputLayer) = inputLayer;
			}
		}
		TB->startingEpoch++;
	}
return TB;
}

void printTrainedNetwork(Brain* TB){
	int i,j;
	double iw;
	printf("#############################################\n");
	printf("RESULTS\n");
	printf("#############################################\n");
	for(i=0;i<TB->trainSetRows;i++){
		double net = 0.0;
		for(j=0;j<TB->trainSetColumns;j++){
		iw = TB->inputLayer->listOfNeurons[j]->inputWeight;
		net+= iw * trainSet[i][j];
		}
	double ex = tanH(net);
	printf("ARTIFICIAL NETWORK OUTPUT: \t%f\n",ex);
	printf("REAL OUTPUT: \t\t\t%f\n",realSet[0][i]);
	if(i<TB->trainSetRows-1)
		printf("\n");
	}
	printf("#############################################\n");
}

void printTrainedIL(Brain* TB){
	int i;
	printf("#############################################\n");
	printf("TRAINED INPUT LAYER\n");
	printf("#############################################\n");
	for(i=0;i<TB->inputLayer->numberOfNeurons;i++){
		printf("NEURON %d:\nInput Weights:\t%f\n",i,TB->inputLayer->listOfNeurons[i]->inputWeight);
		printf("Output Weights:\t%f\n",TB->inputLayer->listOfNeurons[i]->outputWeight);
		if(i!= TB->inputLayer->numberOfNeurons-1)	
			printf("\n");
	}
	printf("#############################################\n");
	printf("\n\n");
}

void printUntrainedIL(Layer* layer){
	int i;
	printf("#############################################\n");
	printf("UNTRAINED INPUT LAYER\n");
	printf("#############################################\n");
	for(i=0;i<layer->numberOfNeurons;i++){
		printf("NEURON %d:\nInput Weights:\t%f\n",i,layer->listOfNeurons[i]->inputWeight);
		printf("Output Weights:\t%f\n",layer->listOfNeurons[i]->outputWeight);
		if(i!= layer->numberOfNeurons-1)	
			printf("\n");
	}
	printf("#############################################\n");
	printf("\n\n");
}

void printUntrainedOL(Layer* layer){
	int i;
	printf("#############################################\n");
	printf("UNTRAINED OUTPUT LAYER\n");
	printf("#############################################\n");
	for(i=0;i<layer->numberOfNeurons;i++){
		printf("NEURON %d:\nInput Weights:\t%f\n",i,layer->listOfNeurons[i]->inputWeight);
		printf("Output Weights:\t%f\n",layer->listOfNeurons[i]->outputWeight);
		if(i!= layer->numberOfNeurons-1)	
			printf("\n");
	}
	printf("#############################################\n");
	printf("\n\n");
}

Brain* initBrainProperties(Brain* TB){
	TB->startingEpoch = 0;
	TB->maxEpoch = 10;
	TB->numbersOfNeuronsInInputLayer = 8;
	TB->numbersOfNeuronsInOutputLayer = 4;
	TB->trainSetRows = 8;
	TB->trainSetColumns = 4;
	TB->targetError = 0.0;
	TB->learningRate = 1.0;
	return TB;
}

void copyOutputWeights(Brain* TB,double* outputW){
	int i = 0;
	for(i=0;i<TB->inputLayer->numberOfNeurons;i++)
		outputW[i] = TB->inputLayer->listOfNeurons[i]->outputWeight;
}

void copybackOutputWeights(Brain* TB, double* outputW){
	int i = 0;
	for(i=0;i<TB->inputLayer->numberOfNeurons;i++)
		TB->inputLayer->listOfNeurons[i]->outputWeight = outputW[i];
}	

int main(){
	srand (time(NULL));
	Brain* TB = malloc(sizeof(Brain));
	Layer* IL = malloc(sizeof(Layer));
	Layer* OL = malloc(sizeof(Layer));
	double* outputW;
	initBrainProperties(TB);
	IL = initIOL(IL,TB->numbersOfNeuronsInInputLayer);
	OL = initIOL(OL,TB->numbersOfNeuronsInOutputLayer);
	TB->inputLayer = IL;
	TB->outputLayer= OL;
	printUntrainedIL(IL);
	printUntrainedOL(OL);
	outputW = malloc(TB->inputLayer->numberOfNeurons * sizeof(double));
	copyOutputWeights(TB,outputW);
	TB = trainBrain(TB);
	copybackOutputWeights(TB,outputW);
	printTrainedIL(TB);
	printTrainedNetwork(TB);
	return 0;
}

