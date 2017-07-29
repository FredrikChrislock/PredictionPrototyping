import os
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import pandas as pd
import model_genotype

class auto_encoder:
    
    def __init__(self, folder_name):
        # Extract data from files and store them as evenly sized cycles 
        self.folder_name = folder_name

        self.time_series_array = pd.read_csv('%s/interpoData.csv' % (folder_name),
                                             index_col = [0])
        


        # Scale the size of the inputs with some pre-defined scaler array
        self.max_values = self.time_series_array.max(axis = 0)
        self.num_dimensions = len(self.max_values)
        #self.scaler = np.array([np.power(10, np.ceil(np.log10(value))) for value in self.max_values])
        self.scaler = np.array([np.round(value) for value in self.max_values])

        self.new_cycle_idxs = pd.read_csv('%s/tripTimeNew.csv' % (folder_name),
                                            index_col = 0).values
        self.new_cycle_idxs = np.squeeze(self.new_cycle_idxs)

        self.time_series_array = self.time_series_array.values.T

    def _create_encoder(self, scale, distribution):
        distribution = distribution/np.sum(distribution)
        limits = np.cumsum(distribution)[:-1]
        limits = scale*limits
        limits = np.concatenate([[0.0],limits])
        def encoder(value):
            if scale == 0:
                return None
            encoding = np.zeros(len(limits)+1)
            if value < 0:
                encoding[0] = 1.0
                return encoding
            for i in range(1, len(limits)):
                if limits[i-1] <= value < limits[i]:
                    encoding[i] = 1.0
                    return encoding
            encoding[-1] = 1.0
            return encoding
        return encoder
    def _create_array_encoder(self, scaler, ranges):
        encoder_list = [self._create_encoder(scale, distribution) for scale, distribution in zip(scaler, ranges)]

        def array_encoder(time_sample):
            encoded_list = [encode(value) for encode, value in zip(encoder_list, time_sample)]
            slicer = np.concatenate([value for value in encoded_list if value is not None]) 
            return slicer
        return array_encoder
    def encode_time_series(self, scaler, ranges):
        array_encoder = self._create_array_encoder(scaler, ranges)
        return np.array([array_encoder(time_sample) for time_sample in self.time_series_array.T])
    def export_cycles(self, scaler, ranges, cycle_size):
        encoded_time_series = self.encode_time_series(scaler, ranges)
        cycles = []
        # Part data into instances
        previous_cycle_end = 0
        for i in range(len(self.new_cycle_idxs)):
            cycle_end = self.new_cycle_idxs[i]
            if cycle_end-previous_cycle_end < cycle_size:
                previous_cycle_end = cycle_end
                continue
            cycle = encoded_time_series[cycle_end-cycle_size:cycle_end]
            cycles += [cycle.T]
            previous_cycle_end = cycle_end
        return cycles

if __name__ == '__main__':
    config = model_genotype.model_genotype_config()
    spawner = model_genotype.model_genotype()
    encoder = auto_encoder('C:/Users/FRCHR/OneDrive - DNV GL/SummerProject2017/Data/System80/WashedData')
    ranges = [spawner.create_random_multilayer(config.input_min_num_ranges, 
                                               config.input_max_num_ranges,
                                               config.input_min_size_range,
                                               config.input_max_size_range) for _ in range(encoder.num_dimensions)]
    scaler = [scale if rnd.random() < 1 else 0 for scale in encoder.scaler]
    encoded_time_series = encoder.export_cycles(scaler, ranges, 400)

    plt.figure()
    plt.imshow(encoded_time_series[1], cmap='gray')
    plt.show()
    exit()


#config = model_genotype.model_genotype_config()
#spawner = model_genotype.model_genotype()
#encoder = auto_encoder('C:/Users/FRCHR/Desktop/InterpolatedData/TurbinA')

#ts = [[0,1,2,3,4,5,6,7,8,9],
#      [0,1,2,3,4,3,2,1,0,1],
#      [3,7,3,7,3,7,3,7,3,7]]