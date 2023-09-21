import numpy as np


class Neural_Network:
    # Initialize the network
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights, output_layer_weights, learning_rate):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.hidden_layer_weights = hidden_layer_weights
        self.output_layer_weights = output_layer_weights

        self.learning_rate = learning_rate

    # Calculate neuron activation for an input
    def sigmoid(self, input):
        output = 1 / (1 + np.exp(-input))  
        return output

    # Feed forward pass input to a network output
    def forward_pass(self, inputs):
        hidden_layer_outputs = []
        for i in range(self.num_hidden):
            # Calculate the weighted sum, and then compute the final output.
            weighted_sum = 0.
            for j in range(self.num_inputs):
                weighted_sum += self.hidden_layer_weights[j][i] * inputs[j]
            output = self.sigmoid(weighted_sum)
            hidden_layer_outputs.append(output)

        output_layer_outputs = []
        for i in range(self.num_outputs):
            # Calculate the weighted sum, and then compute the final output.
            weighted_sum = 0.
            for j in range(self.num_hidden):
                weighted_sum += self.output_layer_weights[j][i] * hidden_layer_outputs[j]
            output = self.sigmoid(weighted_sum)
            output_layer_outputs.append(output)

        return hidden_layer_outputs, output_layer_outputs

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, inputs, hidden_layer_outputs, output_layer_outputs, desired_outputs):
        # Calculate output layer betas.
        output_layer_betas = desired_outputs - output_layer_outputs #np.zeros(self.num_outputs)
        #print('OL betas: ', output_layer_betas)

        # Calculate hidden layer betas.
        hidden_layer_betas = np.zeros(self.num_hidden)
        for i in range(self.num_hidden):
            new_weight = 0
            for k in range(self.num_outputs):
                new_weight += self.output_layer_weights[i][k] * output_layer_outputs[k] * (1 - output_layer_outputs[k]) * output_layer_betas[k]
            hidden_layer_betas[i] = new_weight 
        #print('HL betas: ', hidden_layer_betas)

        # This is a HxO array (H hidden nodes, O outputs)
        delta_output_layer_weights = np.zeros((self.num_hidden, self.num_outputs))
        # Calculate output layer weight changes.
        for h in range(self.num_hidden):
            for o in range(self.num_outputs):
                delta_output_layer_weights[h][o] = self.learning_rate * hidden_layer_outputs[h] * output_layer_outputs[o] * (1 - output_layer_outputs[o]) * output_layer_betas[o]

        # This is a IxH array (I inputs, H hidden nodes)
        delta_hidden_layer_weights = np.zeros((self.num_inputs, self.num_hidden))
        # Calculate hidden layer weight changes.
        for i in range(self.num_inputs):
            for h in range(self.num_hidden):
                delta_hidden_layer_weights[i][h] = self.learning_rate * inputs[i] * hidden_layer_outputs[h] * (1 - hidden_layer_outputs[h]) * hidden_layer_betas[h]

        # Return the weights we calculated, so they can be used to update all the weights.
        return delta_output_layer_weights, delta_hidden_layer_weights

    def update_weights(self, delta_output_layer_weights, delta_hidden_layer_weights):
        # Update the weights.
        self.output_layer_weights += delta_output_layer_weights
        self.hidden_layer_weights += delta_hidden_layer_weights

    def train(self, instances, desired_outputs, epochs):
        for epoch in range(epochs):
            print('epoch = ', epoch)
            predictions = []
            for i, instance in enumerate(instances):
                hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
                delta_output_layer_weights, delta_hidden_layer_weights, = self.backward_propagate_error(
                    instance, hidden_layer_outputs, output_layer_outputs, desired_outputs[i])
                predicted_class = self.predict([instance])  # TODO!
                predictions.append(predicted_class)

                # We use online learning, i.e. update the weights after every instance.
                self.update_weights(delta_output_layer_weights, delta_hidden_layer_weights)

            # Print new weights
            #print('Hidden layer weights \n', self.hidden_layer_weights)
            #print('Output layer weights  \n', self.output_layer_weights)

            # Print accuracy achieved over this epoch
            correctNum = 0
            for i in range (len(desired_outputs)):
                if(desired_outputs[i][predictions[i]] == 1):
                    correctNum += 1
    
            acc = correctNum / len(desired_outputs)
            print('acc = ', acc)

    def predict(self, instances):
        predictions = []
        for instance in instances:
            hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
            #print(output_layer_outputs)
            predicted_class = np.argmax(output_layer_outputs) #returns, 0, 1, or 2
            predictions.append(predicted_class)
        return predictions