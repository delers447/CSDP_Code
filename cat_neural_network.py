#! /usr/bin/python3

#Written by Dan on 22Sep23
#The goal is to code the feed-forward of a nueral network with following layers
#Input : 8, Hidden : 3, Output : 8

import numpy as np
import math

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

input_matrix = np.zeros(8).reshape(1, 8)
hidden_weights = np.random.rand(24).reshape(8, 3)
hidden_matrix = np.zeros(3).reshape(1, 3)
output_weights = np.random.rand(24).reshape(3, 8)
output_matrix = np.zeros(8).reshape(8, 1)

input_matrix[0,3] = 1
hidden_matrix = np.matmul(input_matrix, hidden_weights)
output_matrix = np.matmul(hidden_matrix, output_weights)
output_matrix = softmax(output_matrix)

matrixes = [("input_matrix", input_matrix),
			("hidden_weights", hidden_weights),
			("hidden_matrix", hidden_matrix),
			("output_weights", output_weights),
			("output_matrix", output_matrix)]

for matrix_name, matrix in matrixes:
	print(f"=== {matrix_name} ===")
	print(matrix)

print(f"The Sum of the Out_Matrix is {output_matrix.sum()}")