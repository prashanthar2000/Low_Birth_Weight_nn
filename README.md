Implementation.
    Data Preprocessing.
        -We have removed Education column since its not much informative(value is either 5 or NaN)
        -Also We have predicted the weight as a function of age.
        -If age not found we have(i.e NaN) it is replaced with mean.
        -And in Residence and Delivery phase column NaN values are handled by replacing them with the mode.
        -In case of BP and HB NaN values are replaced by mean
    
    We are building a basic neural network with 4 layers in total: 1 input layer, 2 hidden layers and 1 output layer. All layers will be fully connected.
    
    Feedforward
        The forward pass consists of the dot operation in NumPy, which turns out to be just matrix multiplication.
        As described we have to multiply the weights by the activations of the previous layer. Then we have to apply the activation function to the outcome.
        To get through each layer, we sequentially apply the dot operation, followed by the sigmoid activation function. 
        In the last layer we use the softmax activation function, since we wish to have probabilities of each class, so that we can measure how well our current forward pass performs.
    
    Backpropagation
        The backward pass is hard to get right, because there are so many sizes and operations that have to align, for all the operations to be successful.
    
    Training (Stochastic Gradient Descent)
        We have defined a forward and backward pass, we choose to use Stochastic Gradient Descent (SGD) as the optimizer to update the parameters of the neural network.
        There are two main loops in the training function. One loop for the number of epochs, which is the number of times we run through the whole dataset, and a second loop for running through each observation one by one. 
    
    
List your hyperparameters
        -Learning rate 0.05 
        -activation function: sigmoid ,
        -hidden layers 2 length 8,7 
        -used SGD for training

Detailed steps to run your files 
	Original dataset in kept in data folder

	python3 preprocessing.py
	running this will preprocess the data and write to file called LBW_dataset_clean.csv in data folder

	python3 Nueral_net.py
	this will make use of the preprocessed data(i.e LBW_dataset_clean.csv)
    
