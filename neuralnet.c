#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <sys/types.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <math.h>
#include <stdatomic.h>



pthread_cond_t backPropWait = PTHREAD_COND_INITIALIZER; //if forwardthreads need to wait for backprop to finish
pthread_cond_t backPropGO = PTHREAD_COND_INITIALIZER; // if backprop can go, or if it needs to wait

//holds the values associated with a data point as it passes through the network
typedef struct dataPoint{
	int dataIndex;//index in data matrix
	int *data; //stores the data
	int realParty; //real political party
	int predictedParty; //predicted party from the network
	double *actInput; //stores the activation function outputs from the input layer
	double *actHidden;//stores the activation function outputs from the hidden layer
	double *actOutput; //stores the activation function outputs from the output layer
	struct dataPoint *next; //points to the next data point in a queue
}dataPoint;

//holds the values shared amongst the threads that construct the network
typedef struct constructThreads{
	atomic_int numWaiting; //number of waiting threads
	atomic_int remainingThreads; //number of remaining threads that haven't terminated
	atomic_int batchReady; //1 if there is a batch ready to backpropogate
	atomic_int batchSize; //holds the batch size (normally minibatch)
	double **weightInput; //weight matrix between input and hidden
	double **weightHidden; //weight matrix between hidden and output
	int **dataMat; //data matrix
	int currData; //current index in the data matrix
	int numData;
	int *availableData; // array that holds 1 if a datapoint is "checked out", 0 if available to take
	atomic_int busyThreads; //holds number of threads passing through the network
	struct dataPoint *completeHead; //head of complete queue
	struct dataPoint *completeTail; //tail of complete queue
	int completeSize; //size of complete queue
	atomic_int backPropReady; //if backprop is ready to go
	int terminate; //1 if the threads need to terminate
	pthread_mutex_t *grabData; //blocks multiple threads from grabbing data
	pthread_mutex_t *dropOffData; //blocks multiple threads from dropping off data
	pthread_mutex_t *beginBackProp; //used to awaken backprop
	pthread_mutex_t *editQueue; //blocks multiple threads from editing queue
}constructThreads;

//holds the values shared amongst threads testing the network
typedef struct testThreads{
	double **weightInput;
	double **weightHidden;
	int **dataMat;
	int currData;
	char *set;
	atomic_int totalData;
	atomic_int sum; //sum of all correct guesses made by the network
	atomic_int dataTested; //number of points tested so far

	pthread_mutex_t *grabData;
}testThreads;

void *forwardProp(void *args);
void *backProp(void *args);
void *testNetwork(void *args);
int initialize(char *filename);
int getData(FILE *file);
int *fillData(char *p, char *votes);
int vote(char v);
int party(char p);
int threadManagement(int **trainData, int **testData, int numTrain, int numTest);
double **getWeight(int numRows, int numCols);
double **getInits(int numRows, int numCols, double init);
int printDataMat(int **data, int numData, int numPoints);
int printdataPoint(int *data, int length);
int printMat(double **mat, int numRows, int numCols);
dataPoint *forward(dataPoint *dp, double **weightHidden, double **weightInput);
double sigmoid(double x);
double threshold(double x);
int printQueue(dataPoint *curr);
double backpropogation(dataPoint *headDP, double **weightInput, double **weightHidden);
double updateHidden(dataPoint *curr, double **weightHidden, int j, double update);
double updateInput(dataPoint *curr, double **weightInput, double **weightHidden, int i, int j, double update);
double inputSum(dataPoint *dp, double **weightMat,int size, char layer, int dest);
double sigmoidPrime(double x);
int freeQueue(dataPoint *curr);
int printVec(double *vec, int len);
dataPoint *makeDP(int *data, int index);
int freeMat(int **mat, int numRows);



int numInput; //number of nodes in input layer
int numHidden = 10; //number of nodes in hidden layer
int numOutput = 1; //number of nodes in output layer
int YEA = 1;
int NAY = 0;
int ABSTAIN = -1;
int DEM = 1;
int REP = 0;
int miniBatch = 16; //size of batch to compute gradient descent on (power of 2)
double TERMINATE_STATE = 0.001; //minimize change in update matrix
int TERMINAL_COUNTER = 1000; //maximum number of iters.
double ALPHA = .5; //learning curve for grad. descent


/*
forwardProp threads continually grab data points and forwardpropogate them, 
storing necessary data as they go. The forwardProp threads stop only when the 
backProp thread is preparing to / going through the network. The termination state 
for the construction threads is when the weight matrices no longer upgate between backpropogations.
*/
void *forwardProp(void *args){
	constructThreads *param = (constructThreads *)args;
	int dataIndex;

//go until termination state isn't reached
while(param->terminate == 0){
	//only one forward prop thread can grab a datapoint at a time to keep from grabbing the same one.
	pthread_mutex_lock(param->grabData);
	//if backprop is ready to go, or the needed datapoint is waiting in the backprop queue, wait for backprop
	while((param->backPropReady==1) || (param->availableData[param->currData]==1)){
		//increment number of waiting threads
		param->numWaiting++;
		//if all the threads are waiting on backprop, backprop
		if(param->numWaiting==5){
			param->batchReady = 1;
			//if there aren't a batch amount of values but backprop needs to go anyway, set batchSize smaller
			if(param->completeSize<=miniBatch){
				param->batchSize = param->completeSize;
			}
			else{
				//set batch size to ideal batch size
				param->batchSize = miniBatch;
			}
			pthread_cond_signal(&backPropGO);
		}
		//in the case that no thread signaled backprop and there are no threads in the network, signal backprop
		if(param->busyThreads==0){
			param->batchReady = 1;
			if(param->completeSize<=miniBatch){
				param->batchSize = param->completeSize;
			}
			else{
				param->batchSize = miniBatch;
			}
			pthread_cond_signal(&backPropGO);
		}
		param->currData+=0;
		param->availableData[param->currData]+=0;
		param->completeSize+=0;
		param->numWaiting+=0;
		//sleep
		pthread_cond_wait(&backPropWait, param->grabData);
		//thread wakes up, decrement amount waiting
		param->numWaiting--;
		//signal any threads still waiting on backprop
	pthread_cond_signal(&backPropWait);
	}
	
	//increment amount of threads currently in the network
	param->busyThreads++;
	//grab data index of next data and increment
	dataIndex = param->currData;
	param->currData = (param->currData +1)%param->numData;
	param->availableData[dataIndex] = 1;
	
	pthread_mutex_unlock(param->grabData);
	

	//create a datapoint to traverse the network
	dataPoint *dp = makeDP(param->dataMat[dataIndex],dataIndex);

	//compute forward propogation on the datapoint
	dp = forward(dp,param->weightHidden,param->weightInput);

	//only one thread can drop off data at a time so as to manage signalling backprop
	pthread_mutex_lock(param->dropOffData);
	//only one thread can edit the queue at once (backprop also edits queue)
	pthread_mutex_lock(param->editQueue);
	//if no elems in the queue, make head and tail
	if(param->completeSize==0){
		param->completeHead = dp;
		param->completeTail = dp;
	}
	else{
		//add data point to queue
		param->completeTail->next = dp;
		param->completeTail = dp;
	}
	param->completeSize++;
	pthread_mutex_unlock(param->editQueue);

	param->busyThreads--;
	//if the queue is big enough to backprop
	if(param->completeSize >= miniBatch){
		//check that backprop isn't done to keep threads from waiting
		if(param->terminate==0){
			param->batchReady = 1;
			param->busyThreads+=0;
			param->backPropReady = 1;
		}	
	}
	//if no threads in the network, backprop is ready, and minibatch is correct size, let backprop go
	if(param->busyThreads==0 && param->backPropReady==1 && param->completeSize >= miniBatch){
		param->numWaiting+=0;
		param->batchReady = 1;
		param->batchSize = miniBatch;
		pthread_cond_signal(&backPropGO);
	}
	pthread_mutex_unlock(param->dropOffData);

}

pthread_cond_signal(&backPropGO);
param->remainingThreads--;
//the last thread clears out any remaining queue 
	if(param->remainingThreads==0){
	freeQueue(param->completeHead);
	}
	pthread_exit(0);
}


/*
The backprop thread computes backpropogation for a thread. The thread can only go
when no other threads are passing through the network. The data points it uses for
backpropogation get "dropped off" after backpropogation and are made available to the other threads.
*/
void *backProp(void *args){

	constructThreads *param = (constructThreads *)args;
	double update;
	dataPoint *head;
	dataPoint *prev;

	/*
	option to have backprop return the total loss over the training variables.
	If so, use counter and avg loss to compute the average loss. Since the gradient descent converges to not
	a global minima, the loss does not converge and the threads will not terminate.
	*/
	double counter = 0;
	double avgloss = 0;
	
	while(param->terminate==0){
		counter++;
		pthread_mutex_lock(param->beginBackProp);
		//while there is not a batch ready to backprop or there are threads in the network, wait
		while(param->batchReady==0|| param->busyThreads>0){
			pthread_cond_wait(&backPropGO,param->beginBackProp);
			
		}
		//secure access to the queue, so one thread is not adding to it at the same time 
		//as backprop is dissecting it
		pthread_mutex_lock(param->editQueue);
		head = param->completeHead;

		//dissect the queue so that the complete queue no longer holds the data points
		//to be backpropogated
		for(int i = 0; i<param->batchSize; i++){
			prev = param->completeHead;
			param->completeHead = param->completeHead->next;
			param->completeSize--;
		}

		//if there is no longer a queue large enough to backpropogate, update
		if(param->completeSize<=miniBatch){
			param->batchReady = 0;
		}
		//make the end of the backprop queue point to null
		prev->next = NULL;
		if(param->completeHead==NULL){
			param->completeTail=NULL;
		}
		pthread_mutex_unlock(param->editQueue);

		//backpropogate over the batch
		update = backpropogation(head, param->weightInput, param->weightHidden);
		avgloss+=update;
		
		//"drop off" the data points back at the beginning -- update their availablity and release the datapoint structs
		for(int i = 0; i<param->batchSize; i++){
			prev = head;
			param->availableData[head->dataIndex]=0;
			head = head->next;
			free(prev->actOutput);
			free(prev->actHidden);
			free(prev->actInput);
			prev = NULL;
			free(prev);
		}
		head = param->completeHead;

		//if the weight matrices are no longer updating, or the program has been running too long
		if(update <= TERMINATE_STATE || counter==TERMINAL_COUNTER){
			//terminate the threads
			param->backPropReady = 0;
			param->terminate = 1;
			head = param->completeHead;
			
		}
		param->backPropReady = 0;
		//signal any waiting forward prop threads indicating thei can go
		pthread_cond_signal(&backPropWait);
		pthread_mutex_unlock(param->beginBackProp);

	}
	param->terminate = 1;
	param->backPropReady = 0;
	//signal any other waiting threads
	pthread_cond_signal(&backPropWait);
	param->remainingThreads--;
	if(param->remainingThreads==0){
	freeQueue(param->completeHead);
	}
	pthread_exit(0);
}


/*
 The testnetwork threads forward propogate all datapoints in their set, calculating which ones are correct predictions and which are false.
*/
void *testNetwork(void *args){
	testThreads *param = (testThreads *)args;
	int totalData;
	int dataIndex;
	int dataTested;
	//go until every data point has been tested
	while(param->dataTested < param->totalData){
		//grab index
		dataIndex = atomic_load(&param->dataTested);
		param->dataTested++;
		if(dataIndex==param->totalData){
			break;
		}
		
		//make a data point
		dataPoint *dp = makeDP(param->dataMat[dataIndex],dataIndex);
		
		//forward propogate the point
		dp = forward(dp,param->weightHidden,param->weightInput);
		//update sum
		param->sum += 1-labs(dp->predictedParty - dp->realParty);
	
		
	}
	dataTested = atomic_load(&param->dataTested);
	totalData = atomic_load(&param->totalData);
	pthread_exit(0);

}

/*
An artificial neural network is a form of artificial intelligence that uses supervised learning to find patterns
in a dataset and make predictions about data.  Nerualnet.c builds a threaded neural network that uses mini batch
gradient descent to compute the back propogation.Mini-batch gradient descent computes the updated weight edges using
the average of multiple data points rather than just one at a time. This lends itself to multithreading because it allows
multiple threads to be passed through the network at once rather than waiting after each one. Done correctly, mini-batch gradient descent 
can speed up even a non-threaded computation by minimizing the time spent backpropogating and directing the gradient towards a 
minimum loss of multiple points rather than just the minimum weight values for a single data point.

6 threads build the neural network. 5 threads forward propogate data through the network, computing their activation
function values, and drop the data points in a queue held at the end of the network before going back to grab another element.
1 thread collects a predetermined amount of data points from the queue of completed data and performs mini-batch gradient
descent on them. Execution stops when the weight matrices no longer update more than a certain amount.

After the network is built on a training dataset, 4 threads test the effectiveness of the network. 3 threads forward propogate
data from the training set, and 1 thread propogates data from the test set. These threads compute the accuracy of the model
with regards to both the training and the test set. This is what is outputted by the system.

This neural network is built around a dataset provided by America Chambers. It takes as input a file containing 430 polititians,
along with their political affiliation and how they voted on 10 different issues. This neural network will attempt to predict
the political party of a data point given how they voted. This network can adjust the number of perceptrons in the hidden layer,
and gets the number of perceptrons in the input layer from the data. However, this network only works to predict an outcome
of size 0; otherwise, the formula for gradient descent changes. Also, this network can only support one hidden layer.
Below is a link to the data set described in more detail.

http://mathcs.pugetsound.edu/~alchambers/cs431/assignments/cs431_hw4/cs431_hw4.html

The network does a very poor job of computing the parties for the data. This is because using mini-batch gradient descent, the 
weight matrices are converging but they aren't converging to a minimum point, which explains why if you change the parameters
to converge when the average loss is below a threshold, it will not converge. I think they are incorrectly converging for one of two reasons:

1) The formulas I am using to compute the gradient are incorrect. I based my formulas off of update equations that we discussed in AI
for stochastic gradient descent. Mini-batch gradient descent should base the update for an edge weight off of the average of the 
gradients for all the data points in the batch.

2)The matrix converges on a local minimum very quickly, where the gradient is then zero and the matrix fails to update. Falling into a local minima 
is less likely to occur using stochastic gradient descent, as a new data point has a strong chance of pointing the gradient towards a different
local / global minima. I used the following links to learn about mini-batch gradient descent.

https://www.cs.cmu.edu/~muli/file/minibatch_sgd.pdf
https://ai.stackexchange.com/questions/20377/what-exactly-is-averaged-when-doing-batch-gradient-descent/20380#20380
https://en.wikipedia.org/wiki/Gradient_descent
https://towardsdatascience.com/gradient-descent-algorithm-and-its-variants-10f652806a3
*/
int main(int argc, char *argv[]){

	if(argc !=2){
		printf("I'm sorry! you chose an invalid number of inputs. The file must take in one input, \n");
		printf("the file name. \n");
		return 1;
	}

	char *filename = argv[1];

	time_t t;
	srand((unsigned) time(&t));
	return initialize(filename);

}

/*
 Open the file and check to see if it is valid.
*/
int initialize(char *filename){
	FILE *file = fopen(filename, "r"); //read-only
	if(file==NULL){
		printf("File not found.\n");
		return 1;
	}
	
	return getData(file);
}

/*

The getData function parses up the file and fills the training and test data set.
The training set gets 90% of the data, and the tes set gets the remaining 10%.
*/
int getData(FILE *file){
	printf("getting data");
	//set aside space to hold the data
	int arraySize = 10;
	int **data = (int **)malloc(arraySize*sizeof(int *));
	ssize_t read;
	char *line;
	size_t len;
	char *trash; // useless term at the beginning of the data
	char *party; //holds the party affiliation
	char *votes; //string of the votes
	int numData = 0;
	do{
		line = NULL;
		len = 0;
		read = getline(&line,&len,file);

		//if you hit the end of the file, break
		if(line[0]=='\n' || line==NULL){
			break;
		}
		trash = strtok(line, "\t");
		party = strtok(NULL, "\t");
		votes = strtok(NULL, "\t");
		if(party==NULL || votes==NULL){
			break;
		}

		//if you run out of space, reallocate more
		if(numData==arraySize){
			arraySize+=arraySize;
			data = (int **)realloc(data,arraySize*sizeof(int *));
		}

		//translate the votes into data points and fill the matrix
		data[numData] = fillData(party, votes);
		numData++;
	}while(read!=-1);

	
	int numTrain = .9*numData;
	int numTest = numData-numTrain;
	//allocate space for the two data matrices
	int **trainData = (int **)malloc(numTrain*sizeof(int *));
	int **testData = (int **)malloc(numTest*sizeof(int *));
	//assign the data to the correct matrix
	for(int i = 0; i<numData; i++){
		if(i<numTrain){
			trainData[i] = data[i];
		}
		else{
			testData[i-numTrain] = data[i];
		}
	}
	//spin the threads
	threadManagement(trainData,testData,numTrain,numTest);
	//free the data points
	freeMat(data, numData);
	free(trainData);
	free(testData);
	return 0;

}

/*
Free every row in the matrix, then free the amtrix itself.
*/
int freeMat(int **mat, int numRows){
	for(int i = 0; i<numRows; i++){
		free(mat[i]);
	}
	free(mat);
	return 0;
}

/*
Translate a string of a voting pattern into a data array.
*/
int *fillData(char *p, char *votes){
	int *data = (int *)malloc(sizeof(int)*(strlen(votes)));
	numInput = strlen(votes)-1;
	for(int i = 0; i<strlen(votes); i++){
		if(votes[i]=='\n'){
			break;
		}
		data[i] = vote(votes[i]); //translate the vote into a data point
	}
	data[strlen(votes)-1] = party(*p); //grab the party affiliaton
	return data;
}

/*
Free every element in a queue of datapoints, given its head
*/
int freeQueue(dataPoint *curr){
	if(curr==NULL){
		return 0;
	}
	dataPoint *prev;
	while(curr!=NULL){
		prev = curr;
		curr = curr->next;
		free(prev->actOutput);
		free(prev->actHidden);
		free(prev->actInput);
		//clear its spot in memory
		prev = NULL;
		free(prev);
	}
	return 0;
}

/*
print an array of ints in a row
*/
int printdataPoint(int *data, int length){
	for (int i = 0; i<length; i++){
		if(data==NULL){
			return 1;
		}
		printf("%d \t",data[i]);
	}
	printf("\n");
	return 0;
}

/*
translate a vote into an integer. In this data, '+' is yea, '-' is nay, and '.' is abstain.
*/
int vote(char v){
	if(v=='+'){
		return YEA;
	}
	if(v=='-'){
		return NAY;
	}
	else{
		return ABSTAIN;
	}
}

/*
	Translate a party affiliation into an integer. In this data, Democrat "D" is 1 and Republican 'R' is 0.
*/
int party(char p){
	if(p=='D'){
		return DEM;
	}
	else{
		return REP;
	}
}

/*
	Print a queue of datapoints given its head
*/
int printQueue(dataPoint *head){
	dataPoint *curr = head;
	while(curr !=NULL){
		printf("%d \t",curr->dataIndex);
		curr = curr->next;
	}
	printf("\n");
	return 0;
}

/*
ThreadManagement spins off all the threads for the network, and deals with them when they return.
I used the following website for information on atomic variables:
https://en.cppreference.com/w/c/atomic/atomic_load
https://en.cppreference.com/w/c/atomic/ATOMIC_VAR_INIT
*/
int threadManagement(int **trainData, int **testData, int numTrain, int numTest){
	//allocate space for all the threads
	pthread_t *threads = (pthread_t*)malloc(sizeof(pthread_t) * 10);
	//allocate space for the arguments to the construction threads
	constructThreads *args = (constructThreads *)malloc(sizeof(constructThreads));
	//get randomized weight matrices between the input and hidden layer, and between the hidden and output.
	args->weightInput = getWeight(numInput,numHidden);
	args->weightHidden = getWeight(numHidden,numOutput);
	args->dataMat = trainData;
	args->currData = 0;
	args->numData = numTrain;

	args->availableData = (int *)malloc(numTrain*sizeof(int));
	for(int i = 0; i<numTrain; i++){
		args->availableData[i] = 0;
	}

	args->busyThreads = 0;
	args->completeHead = NULL;
	args->completeTail = NULL;
	args->completeSize = 0;
	args->backPropReady = 0;
	args->batchSize = 0;
	args->terminate = 0;
	args->numWaiting = 0;
	args->batchReady = 0;
	args->remainingThreads = 6;
	//initialize all of the mutexes
	args->grabData = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
	pthread_mutex_init(args->grabData, NULL);
	args->editQueue = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
	pthread_mutex_init(args->editQueue, NULL);
	args->dropOffData =(pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
	pthread_mutex_init(args->dropOffData, NULL);
	args->beginBackProp = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
	pthread_mutex_init(args->beginBackProp, NULL);

	//spin the 5 forward propogation threads
	for(int i = 0; i<5; i++){
		pthread_create(&threads[i], NULL, forwardProp, (void *)args);
	}
	//spin the backpropogation thread
	pthread_create(&threads[5], NULL, backProp, (void *)args);
	
	//join all the construction threads
	for(int i = 0; i<6; i++){
		pthread_join(threads[i],NULL);
	}

	//build arguments for testing the test and training set
	testThreads *argms[2]; 
	for(int i = 0; i<2; i++){
		argms[i] = (testThreads *)malloc(sizeof(testThreads));
		argms[i]->weightInput = args->weightInput;
		argms[i]->weightHidden = args->weightHidden;
		argms[i]->sum = 0;
		argms[i]->dataTested = 0;
		argms[i]->grabData = args->grabData;
		if(i==0){
			argms[i]->dataMat = args->dataMat;
			argms[i]->set = "training";
			argms[i]->totalData = ATOMIC_VAR_INIT(numTrain);

		}
		else{
			argms[i]->totalData = numTest;
			argms[i]->dataMat = testData;
			argms[i]->set = "test";
		}
		
	}

	//spin the testing forward prop threads on training set
	for(int i = 0; i<3; i++){
		pthread_create(&threads[6+i], NULL, testNetwork, (void *)argms[0]);
	}
	//spin the testing thread on test set
	pthread_create(&threads[9], NULL, testNetwork, (void *)argms[1]);

	//join the rest of the threads
	for(int i = 6; i<10; i++){
		pthread_join(threads[i],NULL);
	}

	//for both sets, report the network's accuracy
	int sum;
	int dataTested;
	for(int i = 0; i<2; i++){
		sum = atomic_load(&argms[i]->sum);
		dataTested = atomic_load(&argms[i]->dataTested);
		printf("Accuracy on %s set = %.2f %% \n",argms[i]->set, ((double)((double)sum / (double)dataTested)*100));
	}

	//destroy all mutexes and conditionals
	pthread_cond_destroy(&backPropWait);
	pthread_cond_destroy(&backPropGO);
	pthread_mutex_destroy(args->grabData);
	pthread_mutex_destroy(args->editQueue);
	pthread_mutex_destroy(args->dropOffData);
	pthread_mutex_destroy(args->beginBackProp);

//free data structures
free(args->weightInput);
free(args->weightHidden);
free(args->availableData);
free(threads);
free(args);
free(argms[0]);
free(argms[1]);


return 0;
}

/*
Forward takes a data point and forward propogates it through the network, filling in the 
activation function values as it goes. The activation function in the hidden layer is 
the sigmoid function. To calculate the predicted result, the output layer uses a threshold function.
*/
dataPoint *forward(dataPoint *dp, double **weightHidden, double **weightInput){
	
	//input layer just holds the original data
	for(int i = 0; i<numInput; i++){
		dp->actInput[i] = (double) dp->data[i];
	}

	//hidden layer takes as input the sum of the output from every perceptron in the input layer
	// multiplied by the weight connecting it to each node in the hidden layer
	double sum;
	for(int j = 0; j<numHidden; j++){
		sum = 0;
		//collect the sum
		for(int i = 0; i<numInput; i++){
			sum += weightInput[i][j]*dp->actInput[i];
		}
		//pass it through the sigmoid function
		dp->actHidden[j] = sigmoid(sum);
	}

	//for each perceptron in the output layer
	for(int k = 0; k<numOutput; k++){
		sum = 0;
		//sum over its inputs
		for(int j = 0; j<numHidden; j++){
			sum += weightHidden[j][k]*dp->actHidden[j];
		}
		//pass it through the sigmoid function
		dp->actOutput[k] = sigmoid(sum);
		//the actual prediction(not differentiable) passes this value through a threshold
		dp->predictedParty = threshold(dp->actOutput[k]);
	}
	return dp;
}
/*
print an array of doubles
*/
int printVec(double *vec, int len){
	for(int i = 0; i<len; i++){
		printf("%f \n",vec[i]);
	}
	printf("\n");
	return 0;
}

/*
compute the sigmoid function value for a given input x
*/
double sigmoid(double x){
	return (1/(1+exp(-1*x)));
}

/*
Compute the threshold cuntion for an input x with the threshold value .625 (the y value of the sigmoid function at .5)
*/
double threshold(double x){
	if(x<=.625){
		return 0;
	}
	else{
		return 1;
	}
}


/*
Backpropogation updates the weights for every edge in the network once. It does this using mini-batch gradient descent.
Currently, backpropogation returns the cummulative amount the edges were updated between iterations. 
BY uncommenting the code below, changing the return value and the test case in the backprop thread, the network can
base its construction around capping the average loss over all datapoints. However, since this network does not converge
on a global minima, the loss does not converge and this will lead to the program running continually.
*/
double backpropogation(dataPoint *headDP, double **weightInput, double **weightHidden){
	double update = 0;
	dataPoint *head = headDP;
	
	//for every perceptron in the hidden layer, update the edge connecting it to the output with respect to
	//every node in the queue
	for(int j = 0; j<numHidden; j++){
		update = updateHidden(head, weightHidden,j,update);
	}
	

	//for every perceptron in the input layer, update the edge connecting it to the hidden with respect to
	//every node in the queue
	for(int i = 0; i<numInput; i++){
		for(int j = 0; j<numHidden; j++){
			update = updateInput(head, weightInput, weightHidden, i, j, update);
		}	
	}

/*
	double loss = 0;
	double correct = 0;
	while(headDP!=NULL){
		forward(headDP,weightHidden,weightInput);
		correct +=1-abs(headDP->realParty - headDP->predictedParty);
		loss += fabs(((double)headDP->realParty) - headDP->actOutput[0]);
		//printf("real = %d predicted = %d, output = %f \n",headDP->realParty, headDP->predictedParty, headDP->actOutput[0]);
		headDP = headDP->next;
	}
	headDP = head;
	//printf("backprop train: correct = %f, avg. loss = %f\n",correct,((double)loss/(double)miniBatch));
	*/

	//return ((double)loss/(double)miniBatch);
	return update;
}

/*
update the weight of an edge between the hidden layer and the output layer 
by computing the average gradient over all data points in the queue
*/
double updateHidden(dataPoint *curr, double **weightHidden, int j, double update){

	double temp = weightHidden[j][0]; //grab current value
	double gradient = 0;
	double functionSum;
	dataPoint *head = curr;

	//for every data point
	while(curr!=NULL){
		//compute the sum into the output vector from every perceptron in the hidden layer
		functionSum = inputSum(curr,weightHidden,numHidden, 'h',0);

		//increment the total gradient by the gradient for a single point
		gradient+=((curr->realParty - curr->predictedParty)*sigmoidPrime(functionSum)*curr->actHidden[j]);
		curr = curr->next;
	}
	//update the weight. Note that dividing by miniBatch averages the gradient
	weightHidden[j][0] += ALPHA*((double)(-2/(double)miniBatch))*gradient;
	double change = fabs(weightHidden[j][0]-temp);
	//increment total update
	update += change;
	curr = head;
	return update;
}


/*
update the weight of an edge between the input layer and the hidden layer by computing the average
gradient over all data points in the queue
*/
double updateInput(dataPoint *curr, double **weightInput, double **weightHidden, int i, int j, double update){
	double temp = weightInput[i][j]; //grab initial weight
	double gradient = 0;
	double outputSum;
	double hiddenSum;
	dataPoint *head = curr;

	//over every data point in the queue
	while(curr!=NULL){
		//compute the sum of all inputs into the output node
		outputSum = inputSum(curr, weightHidden,numHidden,'h',0);
		//compute the sum of all inputs into the hidden node j
		hiddenSum = inputSum(curr,weightInput,numInput,'i',j);

		//increment the total gradient by the gradient for a single point
		gradient += ( (curr->realParty - curr->predictedParty)*sigmoidPrime(outputSum)*sigmoidPrime(hiddenSum)*weightHidden[j][0]*curr->actInput[i] );
		curr = curr->next;
	}
	
	//update the weight
	weightInput[i][j] +=ALPHA*((double)(-2/(double)miniBatch))*gradient;
	double change = fabs(weightInput[i][j]-temp);
	//increment total amount updated
	update += change;
	curr = head;
	return update;
}

/*
inputSum computes the sum of all inputs into a specific node in the network.
dest holds the index of the destination node. 
layer specifies which layer is sending the values.
this calculation is specific to the activation values held in a specific data point.
*/
double inputSum(dataPoint *dp, double **weightMat,int size, char layer, int dest){
	double sum = 0;
	double *act;

	//find if it is shipping from the hidden layer or from the input layer
	if(layer=='h'){
		act = dp->actHidden;
	}
	else{
		act = dp->actInput;
	}

	//sum over all activationVal*weight
	for(int i = 0; i<size; i++){
		sum+= act[i]*weightMat[i][dest];
	}
	return sum;
}

/*
sigmoid prime is the prime of the activation function. This is used to calculate gradient descent.
*/
double sigmoidPrime(double x){
	//if the deonimator is invalid
	if(x==0){
		return 0;
	}
	return (exp(-1*x)/((1-exp(-1*x))*(1-exp(-1*x))));
}


/*
create a matrix with random entries between .5 and 1.
*/
double **getWeight(int numRows, int numCols){
	double MAX_NUM = 0.5;

	//allocate space for the matrix
	double **weight = (double **)malloc(numRows*sizeof(double *));
	for(int i = 0; i<numRows; i++){
		weight[i] = (double *)malloc(numCols*sizeof(double));
	}

	//fill each slot with a random number between 0 and 1
	for(int i = 0; i<numRows; i++){
		for(int j = 0; j<numCols; j++){
			weight[i][j] = (double)trunc(100*((double)((rand()/(double)RAND_MAX)*MAX_NUM)+0.5))/100;
		}
	}
	return weight;
}


/*
create a matrix filled with value specified by init.
*/
double **getInits(int numRows, int numCols, double init){
	//allocate space for the matrix
	double **weight = (double **)malloc(numRows*sizeof(double *));
	for(int i = 0; i<numRows; i++){
		weight[i] = (double *)malloc(numCols*sizeof(double));
	}

	//fill each slot with the value init
	for(int i = 0; i<numRows; i++){
		for(int j = 0; j<numCols; j++){
			weight[i][j] = init;
		}
	}
	return weight;
}

/*
print every data point and its data vector
*/
int printDataMat(int **data, int numData, int numPoints){
	for(int i = 0; i<numData; i++){
		printf("data point %d:\n",i);
		for(int j = 0; j<numPoints; j++){
			printf("%d\t",data[i][j]);
		}
		printf("\n");
	}
	return 0;
}

/*
print a matrix of doubles
*/
int printMat(double **mat, int numRows, int numCols){
	for(int i = 0; i<numRows; i++){
		for(int j = 0; j<numCols; j++){
			printf("%f \t",mat[i][j]);
		}
		printf("\n");
	}
	return 0;
}


/*
create a data point out of the data array passed in
*/
dataPoint *makeDP(int *data, int index){
	//allocate space for the datapoint
	dataPoint *dp = (dataPoint *)malloc(sizeof(dataPoint));
	dp->dataIndex = index;
	dp->data = data;
	dp->realParty = dp->data[10];

	//allocate space for the activation vectors for every later
	dp->actInput = (double *)malloc(numInput*sizeof(double));
	dp->actHidden = (double *)malloc(numHidden*sizeof(double));
	dp->actOutput = (double *)malloc(numOutput*sizeof(double));
	dp->next = NULL;
	return dp;
}



