import math
import numpy as np

def sigmoid(x):
    return (1/(1+math.exp(-x)))


learning_rate = 1
weights1 = np.array([1., 1., 1.])
weights2 = np.array([1., 1., 1.])
weights3 = np.array([1., 1., 1.])

print("\tRound 1, inputs 0 0 target 1:")
inputs_r1 = np.array([0, 0])
t1 = 1

#hidden 
number_hidden_layers = 1

for i in range(0, number_hidden_layers):
  hidden_layer_num_nodes = 2
  hidden_layer_outputs = np.zeroes(hidden_layer_num_nodes)
  for j in range (0, len(hidden_layer_outputs)):
    hidden_layer_outputs[j] = np.dot(np.append(inputs_r1, 1), weights2)
    print("Added up the weights for node {}:".format(j), hidden_layer_outputs[j])
    hidden_layer_outputs[j] = sigmoid(hidden_layer_outputs[j])
    print("After activaction function for node {}:".format(j), hidden_layer_outputs[j])


    # net3 = np.dot(np.append(inputs_r1, 1), weights3)
    # print("Added up the weights for node 3:", net3)
    # net3 = sigmoid(net3)
    # print("After activaction function for node 3:", net3)

#output layer

inputs_node1 = hidden_layer_outputs

net1 = np.dot(np.append(inputs_node1, 1), weights1)
print("Added up the weights for node 1:", net1)
net1 = sigmoid(net1)
print("After activaction function for node 1:", net1)

d1 = (t1 - net1) * net1 * (1 - net1)
print("lowercase delta 1:", d1)

deltaW_1 = [
    learning_rate * net2 * d1, learning_rate * net2 * d1,
    learning_rate * 1 * d1
]

print("Change in weights for node1:", deltaW_1)
# weights1 += deltaW_1

# oj (1 - oj) k wjk = o2 (1 - o2) 1 w21 = .731 ( 1 - .731) (.00575 * 1) = .00113
d2 = net2 * (1 - net2) * (d1 * weights1[0])
print("lowercase delta 2:", d2)
deltaW_2 = [
    learning_rate * d2 * inputs_r1[0], learning_rate * d2 * inputs_r1[1],
    learning_rate * d2 * 1
]
print("Change in weights for node2:", deltaW_2)

d3 = net3 * (1 - net3) * (d1 * weights1[1])
print("lowercase delta 3:", d2)
deltaW_3 = [
    learning_rate * d3 * inputs_r1[0], learning_rate * d3 * inputs_r1[1],
    learning_rate * d3 * 1
]
print("Change in weights for node3:", deltaW_3)

#update weights
weights1 += deltaW_1
weights2 += deltaW_2
weights3 += deltaW_3

print("new weights:", weights1, weights2, weights3)



# print()
# print("\tRound 1, inputs 0 1 target 0:")
# inputs_r2 = np.array([0, 1])
# t2 = 0

# #hidden layer
# net2 = np.dot(np.append(inputs_r2, 1), weights2)
# print("Added up the weights for node 2:", net2)
# net2 = sigmoid(net2)
# print("After activaction function for node 2:", net2)

# net3 = np.dot(np.append(inputs_r2, 1), weights3)
# print("Added up the weights for node 3:", net3)
# net3 = sigmoid(net3)
# print("After activaction function for node 3:", net3)

# #output layer

# inputs_node1 = [net2, net3]

# net1 = np.dot(np.append(inputs_node1, 1), weights1)
# print("Added up the weights for node 1:", net1)
# net1 = sigmoid(net1)
# print("After activaction function for node 1:", net1)

# d1 = (t2 - net1) * net1 * (1 - net1)
# print("lowercase delta 1:", d1)

# deltaW_1 = [
#     learning_rate * net2 * d1, learning_rate * net2 * d1,
#     learning_rate * 1 * d1
# ]

# print("Change in weights for node1:", deltaW_1)
# # weights1 += deltaW_1

# # oj (1 - oj) k wjk = o2 (1 - o2) 1 w21 = .731 ( 1 - .731) (.00575 * 1) = .00113
# d2 = net2 * (1 - net2) * (d1 * weights1[0])
# print("lowercase delta 2:", d2)
# deltaW_2 = [
#     learning_rate * d2 * inputs_r2[0], learning_rate * d2 * inputs_r2[1],
#     learning_rate * d2 * 1
# ]
# print("Change in weights for node2:", deltaW_2)

# d3 = net3 * (1 - net3) * (d1 * weights1[1])
# print("lowercase delta 3:", d2)
# deltaW_3 = [
#     learning_rate * d3 * inputs_r2[0], learning_rate * d3 * inputs_r2[1],
#     learning_rate * d3 * 1
# ]
# print("Change in weights for node3:", deltaW_3)

# #update weights
# weights1 += deltaW_1
# weights2 += deltaW_2
# weights3 += deltaW_3

# print("new weights:", weights1, weights2, weights3)
# print()
