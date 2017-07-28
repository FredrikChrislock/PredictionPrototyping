from __future__ import print_function, division
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from datetime import datetime
import numpy as np
import numpy.random as nprnd
import tensorflow as tf
import matplotlib.pyplot as plt
import rnn_network as rnn
import model_genotype as gen

# Spawn initial population
config = gen.model_genotype_config()
spawner = gen.model_genotype()

current_generation = [spawner.spawn_genome(config) for _ in range(config.population_size)]

# For some predifined criterion; train, evaluate, select for sexual reproduction, mate and spawn new population
scores = np.zeros(shape=[config.num_generations, config.population_size])

for generation in range(config.num_generations):
    print("%s: \t Generation %i" % (datetime.now().strftime('%X'), generation))
    generation_scores = np.zeros(config.population_size)
    for ind in range(len(current_generation)):
        individual = current_generation[ind]
        print("%s: \t New individual spawning... " % (datetime.now().strftime('%X')))
        dataset = rnn.LSTM_Dataset(folder_name = "P:/StudentSummerProject2017/SummerProject_OTS/Compiled/26.07/clean/TurbinB", 
                           min_cycle_size = 2400, 
                           target_ranges = individual['target_ranges'], 
                           reuse_factor = 30, 
                           window_size = individual['window_size'])
        graph = tf.Graph()
        with graph.as_default():
            network = rnn.LSTM_Network(input_size = dataset.input_dimension,
                                   output_size = dataset.output_dimension,
                                   num_steps = individual['window_size'],
                                   batch_size = config.batch_size,
                                   lstm_layers = individual['lstm_layers'],
                                   ffwd_layers = individual['ffwd_layers'],
                                   learning_rate = config.max_learning_rate)
        print('%s: \t Spawned.' % (datetime.now().strftime('%X')))
        input, target, predictions, loss, trainer = network.lstm_model()
        with tf.Session(graph=graph) as sess:
            print('%s: \t Initializing graph variables...'% (datetime.now().strftime('%X')))
            sess.run(tf.global_variables_initializer())
            print("%s: \t Starting training..." % (datetime.now().strftime('%X')))
            for e in range(config.num_epochs):
                for i in range(config.num_batches):
                    batch = dataset.draw_train_batch(config.batch_size)
                    _predictions, _loss, _trainer = sess.run([predictions, loss, trainer], feed_dict= {input: batch[0],
                                                                                                      target: batch[1]})
                    #print("%s: \t %i'th iteration. Loss: %f" % (datetime.now().strftime('%X'), i, _loss))
                # Test with one batch
                batch = dataset.draw_test_batch(config.batch_size)
                _predictions, _loss = sess.run([predictions, loss], feed_dict= {input: batch[0],
                                                   target: batch[1]})
                print("%s: \t %i'th epoch. Loss: %f" % (datetime.now().strftime('%X'), e+1, _loss))

            for i in range(config.batch_size):
                plt.figure("Test #%i" % (i))
                num_classes = len(individual['target_ranges'])
                for j in range(num_classes):
                    plt.subplot(num_classes, 1, j+1)
                    plt.plot(_predictions[i,:,j])
                    plt.plot(batch[1][i,:,j])
            plt.show()

            generation_scores[ind] = _loss
            scores[generation, ind] = _loss


            print('%s: \t Session closed' % (datetime.now().strftime('%X')))
        # Choose the top 40% models for the next generation
        winner_idxs = np.argpartition(1-generation_scores, -4)[-4:]
        next_generation = []
        for x in range(int(config.population_size/2)):
            parents = nprnd.choice(winner_idxs, 2)
            parent0 = current_generation[parents[0]]
            parent1 = current_generation[parents[1]]
            child0, child1 = spawner.mate_genomes(config, parent0, parent1)
            child0 = spawner.mutate_genome(config, child0)
            child1 = spawner.mutate_genome(config, child1)
            next_generation += [child0, child1]
        current_generation = next_generation

print('Done')

