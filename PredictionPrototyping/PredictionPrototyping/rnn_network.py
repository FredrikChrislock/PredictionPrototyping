from __future__ import print_function, division

import numpy as np

import numpy.random as rnd

import tensorflow as tf

import matplotlib.pyplot as plt

import network_adapter as net_adapter

import model_genotype as genotype


""" 
Class for storing and delivering training sets, test sets and validation sets to the neural network
"""
class LSTM_Dataset:
    def __init__(self, folder_name, min_cycle_size, target_ranges, reuse_factor, window_size):
        adapter = net_adapter.network_adapter(folder_name = folder_name,
                                              window_size = min_cycle_size)
        self.target_ranges = target_ranges
        self.input_dimension = adapter.data_dimension
        self.output_dimension = len(target_ranges)
        # Define scaling factors for evaluation. Class scores should be attenuated proportionally to the span of the class
        self.class_spans = [x[1]-x[0] for x in target_ranges]
        lead_time = target_ranges[-1][1]
        phase = min_cycle_size - lead_time
        # Generate base target
        base_target = np.zeros(shape = (self.output_dimension, min_cycle_size))
        for i in range(len(target_ranges)):
            _range = target_ranges[i]
            base_target[i,phase+_range[0]:phase+_range[1]] = np.ones(shape=(1,_range[1]-_range[0]))

        # Sample random segments from each cycle 
        max_start_index = min_cycle_size-window_size
        cycles = adapter.cycles
        self.num_cycles = len(cycles)
        train_or_test = ['Train' for _ in range(int(np.ceil(self.num_cycles/2)))] + ['Test' for _ in range(int(np.floor(self.num_cycles/2)))]
        train_or_test = rnd.permutation(train_or_test)
        self.training_set = []
        self.testing_set = []
        for i in range(self.num_cycles):
            cycle = cycles[i]
            for _ in range(reuse_factor):
                idx = rnd.randint(0, max_start_index)
                sample = cycle[idx:idx+window_size]
                sample = np.array(sample, dtype=np.float32)

                target = base_target[:,idx:idx+window_size]
                if train_or_test[i] == 'Train':
                    self.training_set += [(sample, target)]
                elif train_or_test[i] == 'Test':
                    self.testing_set += [(sample, target)]

    def draw_train_batch(self, batch_size):
        input_batch = []
        target_batch = []
        for _ in range(batch_size):
            idx = rnd.randint(0, len(self.training_set))
            input_batch += [self.training_set[idx][0]]
            target_batch += [self.training_set[idx][1]]
        input_batch = np.stack(input_batch)
        target_batch = np.stack(target_batch)
        target_batch = np.transpose(target_batch, [0, 2, 1])
        return (input_batch, target_batch)

    def draw_test_batch(self, batch_size): 
        input_batch = []
        target_batch = []
        for _ in range(batch_size):
            idx = rnd.randint(0, len(self.testing_set))
            input_batch += [self.testing_set[idx][0]]
            target_batch += [self.testing_set[idx][1]]
        input_batch = np.stack(input_batch)
        target_batch = np.stack(target_batch)
        target_batch = np.transpose(target_batch, [0, 2, 1])
        return (input_batch, target_batch)

    #def evaluate_batch(self, predictions, targets):
        # Any target class gets values dependent on two things:
        # 1. How long from a trip the class is ([0,1) proportional with max_lead_time)
        # 2. How wide a range the class covers ([0,1) inversely proportional with the covering ratio for the class






class LSTM_Network:
    def __init__(self, input_size, output_size, num_steps, batch_size, lstm_layers, ffwd_layers, learning_rate):
        
        print('Loading: ', end="", flush=True)
        # Generate the base model
        self.input_tensor = tf.placeholder(dtype=tf.float32, shape = [batch_size, num_steps, input_size], name="data_input")
        _lstm_layers = [self.lstm_cell(x) for x in lstm_layers]
        self.stacked_lstm = tf.contrib.rnn.MultiRNNCell(_lstm_layers)
        self.initial_state = self.state = self.stacked_lstm.zero_state(batch_size, tf.float32)
        _ffwd_layers = []
        previous = lstm_layers[-1]
        for layer in ffwd_layers:
            _w = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[previous, layer]))
            _b = tf.Variable(tf.constant(0.0, shape=[1, layer], dtype=tf.float32))
            _ffwd_layers += [(_w,_b)]
            previous = layer
        _w = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[previous, output_size]))
        _b = tf.Variable(tf.constant(0.0, shape=[1, output_size], dtype=tf.float32))
        _ffwd_layers += [(_w,_b)]
        _ffwd_array = [[None for _ in range(batch_size)] for _ in range(len(_ffwd_layers))]
        self.predictions = []
        print('##', end="", flush=True)
        for step in range(num_steps):
            output, self.state = self.stacked_lstm(self.input_tensor[:,step,:], self.state)
            for j in range(len(_ffwd_array[0])):
                _output = output[j]
                _output = tf.expand_dims(_output,0)
                _ffwd_array[0][j] = tf.matmul(_output, _ffwd_layers[0][0]) + _ffwd_layers[0][1]
            for i in range(1, len(_ffwd_array)):
                layer = _ffwd_array[i]
                for j in range(len(layer)):
                    _ffwd_array[i][j] = tf.matmul(_ffwd_array[i-1][j], _ffwd_layers[i][0]) + _ffwd_layers[i][1]
            self.predictions += [_ffwd_array[-1][:]]
        print('##', end="", flush=True)
        self.predictions = [tf.stack(sequence) for sequence in self.predictions]
        self.predictions = tf.stack(self.predictions)
        self.predictions = tf.transpose(self.predictions, [1, 0, 2, 3])
        self.predictions = tf.squeeze(self.predictions)
        print('##', end="", flush=True)
        # Generate loss-function
        self.target_output = tf.placeholder(dtype=tf.float32, shape=(batch_size, num_steps, output_size), name="target_output")

        self.loss = tf.losses.mean_squared_error(self.target_output, self.predictions) 
        print('##', end="", flush=True)
        # Generate trainer
        self.optimizer = tf.train.AdamOptimizer(learning_rate)

        print('##', end="", flush=True)
        self.trainer = self.optimizer.minimize(self.loss)
        
        print('##', flush=True)

    def lstm_cell(self, num_units):

        return tf.contrib.rnn.BasicLSTMCell(num_units=num_units, state_is_tuple=True)

    def lstm_model(self):

        return self.input_tensor, self.target_output, self.predictions, self.loss, self.trainer

    



if __name__== "__main__":
    config = genotype.model_genotype_config()
    spawner = genotype.model_genotype()
    individual = spawner.spawn_genome(config)
    dataset = LSTM_Dataset(folder_name = "P:/StudentSummerProject2017/SummerProject_OTS/InterpolatedData/TurbinB/", 
                           min_cycle_size = 60*24*7, 
                           target_ranges = individual['target_ranges'], 
                           reuse_factor = 500, 
                           window_size = individual['window_size'])

    batch_size = 4
    batch = dataset.draw_batch(batch_size)

    network = LSTM_Network(input_size = dataset.input_dimension,
                           output_size = dataset.output_dimension,
                           num_steps = individual['window_size'],
                           batch_size = batch_size,
                           lstm_layers = individual['lstm_layers'],
                           ffwd_layers = individual['ffwd_layers'],
                           learning_rate = config.max_learning_rate)

    exit()