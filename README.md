# Political-neural-net

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
