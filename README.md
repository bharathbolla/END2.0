# END2.0

## Assignment-3
### Following is the task

![image](https://user-images.githubusercontent.com/10822997/119270403-202e2a00-bc1a-11eb-90f1-f1b98ee696cd.png)


### Startegy
### Data Representation
The image data is MNSIST data at 28x28
The random number input data is in one hot encoded format
### Data Generation Strategy
Wrote a custom data loader that generates the random interge on the fly
After generating the random number, it was added to the mnist label to generate second label
The input random number is encoded into one hot encoding format
A list comrehension was used to apply the one hot encoding on the entire batch
Batch size of 32 was used

### How the inputs are combined
After running the convnet model on mnist data, and flatten layer has been applied to generate a vector of size 10.
to this vector , the one hot encoded random number input was concatenated using torch.cat
Two linear layers are applied on mnist flattened output and combined input separetly to generate two 10 size vectors

### Results evaluation (Loss function)
Both the results evaluated using Cross entriopy Loss Function

### What results you finally got and how did you evaluate your results
Ran the model with SGD as optimizer for 10 epochs.
Accuracy of the network on the 10,000 test images:  98.47244408945687
Accuracy of the network on the 10,000 test random number input:  98.18290734824281
Results where also evaluated by looking at a single batch.

### What loss function you picked and why!
Categorical cross entropy is picked as loss for function for both the outputs as the outputs in both the cases are fixed number of classes 
and we are predicting one of the class

### What are your learnings
1. Writing custom data loader
2. Combining two inputs which is basis for multi modal architecture
3. Picking the loss function
4. Generating one hot encoded random numbers and applying the function on a batch
5. Debugging the model inputs and outputs at each stage
6. Spent hours figuring out the CUDA error but it's just basically an error in not getting the dimension of the second output as 19


![image](https://user-images.githubusercontent.com/10822997/119270898-9cc20800-bc1c-11eb-97bb-7b6955bed3ff.png)

