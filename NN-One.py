import numpy as np
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt
import random

file_prefix = 'Set of groups'
shapes = ['Circle', 'Ellipse', 'triangle']
groups_of_train = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
groups_of_test = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
make_random = True

learning_rate = 0.5
epochss = 150
# Layers
input_neurons_layer = 100*100  # For a 100X100 matrix # Image size
hidden_neurons_layer = 50  # Amount of neurons in the hidden layer
output_neurons_layer = 3  # 3 because we have 3 different shapes (Triangle, Circle and Ellipse)

files_list_of_train = []
files_list_of_test = []

for group in groups_of_train:
    for shape in shapes:
        filename = f"{file_prefix}_group_{group}_shape_of_{shape}.png"
        files_list_of_train.extend([filename])

for group in groups_of_test:
    for shape in shapes:
        filename = f"{file_prefix}_group_{group}_shape_of_{shape}.png"
        files_list_of_test.extend([filename])

if make_random==True:
    random.shuffle(files_list_of_train)
    random.shuffle(files_list_of_train)


def activation_fun(value):
    return 1 / (1 + np.exp(-value))


def activation_derivative(value):
    return value * (1 - value)


# Function to load and preprocess images
# NN structure

# Weights
weights_from_input_to_hidden = np.random.uniform(
    size=(input_neurons_layer, hidden_neurons_layer)) /input_neurons_layer # The weight from the input layer to the hidden layer
weights_from_hidden_to_output = np.random.uniform(
    size=(hidden_neurons_layer, output_neurons_layer)) /hidden_neurons_layer # The weight from the hidden layer to the output layer
# Bias
bias_hidden = np.random.uniform(size=(1, hidden_neurons_layer)) / input_neurons_layer  # Ensure the shape matches the hidden layer
bias_output = np.random.uniform(size=(1, output_neurons_layer))  /hidden_neurons_layer # Ensure the shape matches the output layer


# Feedforward process
# The function processes the entire input data in one go through matrix multiplications and vectorized operations,
# applying the neural network's weights and biases and then the activation function to produce the network's output efficiently.
# In this case the input is a vector of size 100*100 that represent an Image in black and white of a shape
# Where 1 is a black pixel and 0 is a white pixel.

def feedforward(input_Data):
    hidden_layer_activation = np.dot(input_Data,
                                     weights_from_input_to_hidden) + bias_hidden  #z1
    hidden_layer_output = activation_fun(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, weights_from_hidden_to_output) + bias_output
    predicted_output = activation_fun(output_layer_activation)
    return hidden_layer_output, predicted_output  # The function is returning the expected output for a given input


# Backpropagation
def backpropagate(input_Data, hidden_layer_output, predicted_output,
                  actual_Output):  # responsible for updating the weights and biases of the network based on the errors computed during forward pass.
    global weights_from_input_to_hidden, weights_from_hidden_to_output
    global bias_hidden, bias_output

    # Calculate error
    error = actual_Output - predicted_output
    d_predicted_output = error * activation_derivative(predicted_output) #delta y

    # Hidden layer error
    error_hidden_layer = d_predicted_output.dot(weights_from_hidden_to_output.T)
    d_hidden_layer = error_hidden_layer * activation_derivative(
        hidden_layer_output)  # calculates the gradients of the loss function  #delta hidden

    # Update weights and biases
    weights_from_hidden_to_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate

    weights_from_input_to_hidden += input_Data.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate


# Training the neural network
# Epoch is One complete pass through the entire training dataset.
# By iterating through multiple epochs, the neural network progressively adjusts its weights to minimize the loss function,
# ideally improving its performance on the training data over time.
#
def train(input_Data, Actual_output, epochss, learning_rate,loss_list):

    hidden_layer_output, predicted_output = feedforward(
            input_Data)  # The function is returning the expected output for a given input foe the hidden layer and for the final output
    backpropagate(input_Data, hidden_layer_output, predicted_output, Actual_output)

        #if epoch % 100 == 0:  # Printing the loss every 100 epochs helps monitor the training progress and check if the network is learning effectively.
    loss = np.mean(np.square(Actual_output - predicted_output))
    loss_list= np.append(loss_list,loss)

    if np.argmax(predicted_output) == np.argmax(Actual_output):
        correct = True
    else:
        correct = f'False - it is not {shapes[np.argmax(predicted_output)]}'

    print(f'Epoch {epochh} Loss: {loss},file name: "{filename}" ,Accuracy: {correct}')



    return loss_list

def Photo_to_Binary_list(image_path):
    # Check if the file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The file {image_path} does not exist.")

    # Open the image
    img = Image.open(image_path).convert("L")  # Ensure the image is in grayscale
    pixels = img.load()
    width, height = img.size
    # Convert to binary
    binary_photo = [
        [
            int(pixels[x, y] < 125)
            for y in range(height)
        ]
        for x in range(width)

    ]

    photo_vec = []
    for row in binary_photo:
       photo_vec.extend(row)

    return photo_vec

### Train

num_of_photos= np.size(groups_of_train) * 3
loss_list = np.array([])
for epochh in range(epochss):
    if make_random == True:
        random.shuffle(files_list_of_train)
    for filename in files_list_of_train:
        input_data = Photo_to_Binary_list(filename)
        if 'Circle' in filename:
            y = [1, 0, 0]
        if 'Ellipse' in filename:
            y = [0, 1, 0]
        if 'triangle' in filename:
            y = [0, 0, 1]
        loss_list = train(np.array(input_data).reshape(1,len(input_data)), np.array(y), epochh, learning_rate,loss_list)
    num_correct = 0
    for filename in files_list_of_train:
        input_data = Photo_to_Binary_list(filename)
        if 'Circle' in filename:
            y = [1, 0, 0]
        if 'Ellipse' in filename:
            y = [0, 1, 0]
        if 'triangle' in filename:
            y = [0, 0, 1]
        hidden_layer_output, predicted_output = feedforward(input_data)
        if np.argmax(predicted_output) == np.argmax(y):
            num_correct += 1


    percent_of_accuracy= num_correct/num_of_photos
    text = f"The percent of accuracy{percent_of_accuracy}"
    print(text)
    if percent_of_accuracy>0.75:
        text = f"We Break - The percent of accuracy{percent_of_accuracy}"
        print(text)
        break


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed

plt.plot(smooth(loss_list,0.95))
plt.show()



### TEST

num_of_photos=np.size(groups_of_test)*3
num_of_photos=np.size(groups_of_test)*3
num_correct = [0,0,0]
for filename in files_list_of_test:
    input_data = Photo_to_Binary_list(filename)
    if 'Circle' in filename:
        y = [1,0,0]
    if 'Ellipse' in filename:
        y = [0,1,0]
    if 'triangle' in filename:
        y = [0,0,1]
    hidden_layer_output, predicted_output = feedforward(input_data)
    if np.argmax(predicted_output) == np.argmax(y):
        num_correct[np.argmax(y)] += 1


percent_of_accuracy= sum(num_correct)/num_of_photos
percent_of_circle = num_correct[0]/(num_of_photos/3)
percent_of_ellipse = num_correct[1]/(num_of_photos/3)
percent_of_triangle = num_correct[2]/(num_of_photos/3)

text = (f"The percent of accuracy of test group {percent_of_accuracy};\n"
        f"circle: {percent_of_circle}\n"
        f"ellipse: {percent_of_ellipse}\n"
        f"triangle: {percent_of_triangle}")
print(text)


