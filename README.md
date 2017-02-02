# Perceptron-with-early-stopping
An implementation of a Perceptron using early stopping to prevent over-fitting. Used the step function as the activation function.

* The program takes arguments in the form: [train-set][tune-set][test-set].

#Input file format:
1. The files are expected to have the following format:
  * The first line, an integer for the number of features per example.
  * Next, list each feature on its own line with the its possible values, ex. *gender – male female*.
  * Next, the two label features are listed, each on its own line.
  * Next, an integer for the number of examples in the file, on its own line.
  * Lastly, the examples, listed as the example each on its own line, ex.[example  name][label][feature values]. The feature values should have the same order as they were given to the program.
2. The program will ignore empty lines and lines that start with “//”.
3. The program takes binary features and labels. If the set has numerical values, this is easy to work around, just have a threshold for each value.
4. Change the step and patience values according to the desired error value. 

