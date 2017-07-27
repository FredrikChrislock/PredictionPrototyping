from __future__ import print_function, division

import os

import numpy as np

import numpy.random as rnd

import matplotlib.pyplot as plt

import pandas as pd

import csv

class network_adapter:

    def __init__(self, **kwargs):
        # Extract data from files and store them as evenly sized cycles 
        self.folder_name = kwargs['folder_name']
        self.window_size = kwargs['window_size']

        csvInterpoData = csv.reader(open(self.folder_name+"/interpoData.csv"))
        next(csvInterpoData)
        self.interpolated_data = [[float(i) for i in row[1:]] for row in csvInterpoData]
        self.data_dimension = len(self.interpolated_data[0])

        csvTripTimeNew = csv.reader(open(self.folder_name+"/tripTimeNew.csv"))
        next(csvTripTimeNew)
        self.new_cycle_idxs = [int(row[1]) for row in csvTripTimeNew]

        # Part data into instances
        self.cycles = []
        previous_cycle_end = 0
        for i in range(len(self.new_cycle_idxs)):
            cycle_end = self.new_cycle_idxs[i]
            if cycle_end-previous_cycle_end < self.window_size:
                previous_cycle_end = cycle_end
                continue
            cycle = self.interpolated_data[cycle_end-self.window_size:cycle_end]
            self.cycles += [cycle]
            previous_cycle_end = cycle_end
        self.num_cycles = len(self.cycles)

    def initialize_dataset(self, lead_time, num_samples):
        if self.window_size < 4 * lead_time:
            raise ValueError('Lead time can maximum be %i, a fourth if the window size' % (self.window_size / 4))
        self.num_samples = num_samples
        self.lead_time = lead_time
        window_range = 3 * lead_time
        max_start_index = self.window_size-window_range

        self.dataset = []
        for cycle in self.cycles:
            for _ in range(int(num_samples/self.num_cycles)):

                phase_shift = rnd.randint(0, lead_time)
                idx = max_start_index - phase_shift
                sample = cycle[idx:idx+2*lead_time]
                sample = np.array(sample, dtype=np.float32)

                target = [0.0 for _ in range(2*lead_time)]
                for i in range(lead_time-phase_shift):
                    target[-(i+1)] = 1.0
                target = np.array(target, dtype=np.float32)
                target = np.expand_dims(target, axis=-1)
                self.dataset += [(sample, target)]

    def draw_batch(self, batch_size):
        input_batch = []
        target_batch = []
        for _ in range(batch_size):
            idx = rnd.randint(0, len(self.dataset))
            input_batch += [self.dataset[idx][0]]
            target_batch += [self.dataset[idx][1]]
        input_batch = np.stack(input_batch)
        target_batch = np.stack(target_batch)
        return (input_batch, target_batch)

if __name__ == '__main__':
    adapter = network_adapter(folder_name = "C:/Users/FRCHR/OneDrive - DNV GL/SummerProject2017/Data/System80/WashedData/",
                          window_size = 800)
    adapter.initialize_dataset(50, 200)
    batch = adapter.draw_batch(8)

    exit()
