from __future__ import print_function, division
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import numpy.random as nprnd
import matplotlib.pyplot as pyplt
import tensorflow as tf
from rnn_network import LSTM_Network
from network_adapter import network_adapter

print('Loading dataset...')
adapter = network_adapter(folder_name = "P:/StudentSummerProject2017/SummerProject_OTS/InterpolatedData/TurbinB/",
                          window_size = 2400)

adapter.initialize_dataset(12, 2048)
batch_size = 16
num_batches = 20000
print('Loading network...')
network = LSTM_Network(adapter.data_dimension,1,2*adapter.lead_time,batch_size,[64,32,32],[32,16],0.001)
input, target, predictions, loss, trainer = network.lstm_model()

print('Initializing session...')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    losses = []
    true_positives = []
    false_negatives = []
    true_negatives = []
    false_positives = []

    print('Starting trainer...')
    for _ in range(num_batches):
        batch = adapter.draw_batch(batch_size)
        _predictions, _loss, _trainer = sess.run([predictions, loss, trainer], feed_dict= {input: batch[0],
                                                                                           target: batch[1]})

        losses += [_loss]
        it = np.nditer(_predictions, flags=['multi_index'])
        tp_count = 0
        fn_count = 0
        tn_count = 0
        fp_count = 0

        while not it.finished:
            if _predictions[it.multi_index] > 0.5 and batch[1][it.multi_index]==1.0:
                tp_count += 1
            elif _predictions[it.multi_index] <= 0.5 and batch[1][it.multi_index]==0.0:
                fn_count += 1
            elif _predictions[it.multi_index] > 0.5 and batch[1][it.multi_index]==0.0:
                tn_count += 1
            elif _predictions[it.multi_index] <= 0.5 and batch[1][it.multi_index]==1.0:
                fp_count += 1

            it.iternext()
        true_positives += [tp_count/(batch_size*adapter.lead_time*2)]
        false_negatives += [fn_count/(batch_size*adapter.lead_time*2)]
        true_negatives += [tn_count/(batch_size*adapter.lead_time*2)]
        false_positives += [fp_count/(batch_size*adapter.lead_time*2)]

    # Test at the end
    batch = adapter.draw_batch(batch_size)
    _predictions = sess.run([predictions], feed_dict= {input: batch[0],
                                                       target: batch[1]})
    pyplt.figure(0)
    for i in range(6):
        p_sequence = np.squeeze(_predictions[0][i])
        t_sequence = np.squeeze(batch[1][i])
        pyplt.subplot(611 + i)
        pyplt.plot(range(adapter.lead_time*2), p_sequence , range(adapter.lead_time*2), t_sequence)
    pyplt.show()

pyplt.figure(1)
pyplt.subplot(511)
pyplt.plot(range(num_batches),np.array(losses))
pyplt.title('Losses')
pyplt.subplot(512)
pyplt.plot(range(num_batches),np.array(true_positives))
pyplt.title('True positives')
pyplt.subplot(513)
pyplt.plot(range(num_batches), np.array(false_negatives))
pyplt.title('False negatives')
pyplt.subplot(514)
pyplt.plot(range(num_batches), np.array(true_negatives))
pyplt.title('True negatives')
pyplt.subplot(515)
pyplt.plot(range(num_batches), np.array(false_positives))
pyplt.title('False positives')
pyplt.show()

exit()