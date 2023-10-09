#! /usr/bin/python3

#Programmed by dan on 07Oct2023.
#Use skipgram to train a neural network on a corpus.
#Then measure training success by outputting which words are associated with 'cat'.


import numpy as np
import math, random

EPOCHS = 500_001

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x))

def dsigmoid(y):
    #return (sigmoid(x) * sigmoid(1 - x))
    return y *(1 - y)

def softmax(vector):
    output_matrix = np.zeros(vector.size).reshape(vector.shape)
    value_vector = np.zeros(vector.size).reshape(vector.shape)
    sum = 0
    for i, element in enumerate(np.nditer(vector)):
        value_vector[0, i] = math.exp(element)
        sum += math.exp(element)
    for i, element in enumerate(np.nditer(value_vector)):
         output_matrix[0, i] = element / sum
    return output_matrix


def print_results(word_vector, count=None):
    #Word Vector:   1     2       3     4     5       6        7          8
    vector_key = ['the', 'dog', 'saw', 'a', 'cat', 'chased', 'climbed', 'tree']
    #word_vector = softmax(word_vector.transpose()).transpose()
    if count != None:
        print(f"==== {count}th Results ====")
    else:
        print("==== Results ====")
    print("Word \t Result")
    for word, result in zip(vector_key, word_vector):
        print(f"{word}\t {result}")

class NeuralNetwork:
    def __init__(self, numI, numH, numO):
        self.learning_rate = .01
        self.input_nodes = numI
        self.hidden_nodes = numH
        self.output_nodes = numO

        self.weights_ih = np.random.random((self.hidden_nodes, self.input_nodes))
        self.weights_ho = np.random.random((self.output_nodes, self.hidden_nodes))

        self.bias_h = np.random.random((self.hidden_nodes, 1))
        self.bias_o = np.random.random((self.output_nodes, 1))

    def feedforward(self, input_array):
        #LOTS OF MATRIX MATH!
        inputs = np.array(input_array).reshape(self.input_nodes, 1)

        hidden_guess = sigmoid(np.matmul(self.weights_ih, inputs) + self.bias_h)
        output_guess = sigmoid(np.matmul(self.weights_ho, hidden_guess) + self.bias_o)

        return output_guess

    def train(self, inputs_list, answers_list):
        inputs = np.array(inputs_list).reshape(self.input_nodes, 1)
        answers = np.array(answers_list).reshape(self.output_nodes, 1)
        hidden_guess = sigmoid(np.matmul(self.weights_ih, inputs) + self.bias_h)
        output_guess = sigmoid(np.matmul(self.weights_ho, hidden_guess) + self.bias_o)

        #calculate the errors: Error = Answer - Guess
        output_errors = answers - output_guess
        hidden_errors =  np.matmul(self.weights_ho.T, output_errors)

        output_gradient = self.learning_rate * output_errors * dsigmoid(output_guess)
        weights_ho_delta = np.matmul(output_gradient, hidden_guess.transpose())
        self.weights_ho += weights_ho_delta
        self.bias_o += output_gradient

        hidden_gradient = self.learning_rate * hidden_errors * dsigmoid(hidden_guess)
        weights_ih_delta = np.matmul(hidden_gradient, inputs.transpose())
        self.weights_ih += weights_ih_delta
        self.bias_h += hidden_gradient

def main():
    nn = NeuralNetwork(8, 3, 8)

    #Training Coprus
    #“the dog saw a cat”,
    #“the dog chased the cat”,
    #“the cat climbed a tree”.

    #Word Vector:   1     2       3     4     5       6        7          8
    vector_key = ['the', 'dog', 'saw', 'a', 'cat', 'chased', 'climbed', 'tree']

    #Modified Skip-Gram vectorization
    # the   -> dog, saw
    datum1 = ([1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0])
    # dog   -> the, saw, a
    datum2 = ([0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 1, 1, 0, 0, 0, 0])
    # saw   -> the, dog, a, cat
    datum3 = ([0, 0, 1, 0, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 0, 0])
    # a     -> dog, saw, cat
    datum4 = ([0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 1, 0, 1, 0, 0, 0])
    # cat   -> saw, a
    datum5 = ([0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0])
    # the   -> dog, chased
    datum6 = ([1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0])
    # dog   -> the, chased, the
    datum7 = ([0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1, 0, 0])
    # chased-> the, dog, the, cat
    datum8 = ([0, 0, 0, 0, 0, 1, 0, 0], [1, 1, 0, 0, 1, 0, 0, 0])
    # the   -> dog, chased, cat
    datum9 = ([1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 1, 1, 0, 0])
    # cat   -> chased, the
    datum10 = ([0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 1, 0, 0])
    # the   -> cat, climbed
    datum11 = ([1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 1, 0])
    # cat   -> the, climbed, a
    datum12 = ([0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0, 1, 0])
    # climbed> the, cat, a, tree
    datum13 = ([0, 0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 1, 1, 0, 0, 1])
    # a     -> cat, climbed, tree
    datum14 = ([0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 1, 1])
    # tree  -> climbed, a
    datum15 = ([0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 1, 0])

    training_data = [datum1,datum2,datum3,datum4,datum5,
                    datum6,datum7,datum8,datum9,datum10,
                    datum11,datum12,datum13,datum14,datum15]

    the_vector = [1, 0, 0, 0, 0, 0, 0, 0]
    dog_vector = [0, 1, 0, 0, 0, 0, 0, 0]
    saw_vector = [0, 0, 1, 0, 0, 0, 0, 0]
    a_vector = [0, 0, 0, 1, 0, 0, 0, 0]
    cat_vector = [0, 0, 0, 0, 1, 0, 0, 0]
    chased_vector = [0, 0, 0, 0, 0, 1, 0, 0]
    climbed_vector = [0, 0, 0, 0, 0, 0, 1, 0]
    tree_vector = [0, 0, 0, 0, 0, 0, 0, 1]

    for i in range(EPOCHS):
        if i%100_00 == 0:
            print(f"The learning rate is {nn.learning_rate}")
            print_results(nn.feedforward(cat_vector), i)
            nn.learning_rate -= nn.learning_rate * .1
        random.shuffle(training_data)
        for datum in training_data:
            nn.train(datum[0], datum[1])

main()
