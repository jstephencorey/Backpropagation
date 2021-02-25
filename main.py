import math
import numpy as np

# globals:
# number_hidden_layers = 1 # Assuming only one layer
learning_rate = 1
hidden_layer_num_nodes = 2

num_inputs = 2
num_outputs = 1



def sigmoid(x):
    return (1/(1+math.exp(-x)))


weights1 = np.array([1., 1., 1.])
weights2 = np.array([1., 1., 1.])
weights3 = np.array([1., 1., 1.])

# weight_array= np.random.rand(num_inputs+1,num_outputs)*1-0.5
hidden_weight_array = np.ones((hidden_layer_num_nodes, num_inputs + 1))

inputs_array = np.array([[0, 0]])
targets_array = np.array([1])

for k in range (0, len(inputs_array)):
  print("\tRound 1, inputs 0 0 target 1:")

  inputs = inputs_array[k]
  target = targets_array[k]

  #hidden 
  hidden_layer_outputs = np.zeros(hidden_layer_num_nodes)
  for j in range (0, hidden_layer_num_nodes):
    net = np.dot(np.append(inputs, 1), hidden_weight_array[j])
    print("Added up the weights for node {}:".format(j), net)
    hidden_layer_outputs[j] = sigmoid(net)
    print("After activaction function for node {}:".format(j), hidden_layer_outputs[j])


  net2 = hidden_layer_outputs[0]
  net3 = hidden_layer_outputs[1]


  #output layer
  output_layer_weight_array = np.ones((num_outputs, len(hidden_layer_outputs) + 1))

  output_layer_inputs = hidden_layer_outputs

  output_layer_outputs = np.zeros(num_outputs)

  for i in range (0, num_outputs):
    net = np.dot(np.append(output_layer_inputs, 1), output_layer_weight_array[i])
    print("Added up the weights for node 1:", net)
    output_layer_outputs[i] = sigmoid(net)
    print("After activaction function for node 1:", output_layer_outputs[i])

  #Weight adjustment step
  for i in range(0, num_outputs):


    output_delta = (target - output_layer_outputs[i]) * output_layer_outputs[i] * (1 - output_layer_outputs[i])
    print("lowercase delta 1:", output_delta)

    deltaW_1 = []
    for j in range(0, hidden_layer_num_nodes):
      deltaW_1.append(learning_rate * hidden_layer_outputs[j] * output_delta)
    deltaW_1.append(learning_rate * 1 * output_delta)

    print("Change in weights for node1:", deltaW_1)
    output_layer_weight_array[0] += deltaW_1


  for i in range(0, hidden_layer_num_nodes):
    hidden_delta = hidden_layer_outputs[i] * (1 - hidden_layer_outputs[i]) * (output_delta * weights1[0])
    print("lowercase delta 2:", hidden_delta)
    deltaW_2 = []
    for j in range(num_inputs):
      deltaW_2.append(learning_rate * hidden_delta * inputs[j])
    deltaW_2.append(learning_rate * hidden_delta * 1)
    print("Change in weights for node2:", deltaW_2)
    hidden_weight_array[i] += deltaW_2


  print("new weights:", output_layer_weight_array, hidden_weight_array)

    # DONE UP TO HERE!


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

