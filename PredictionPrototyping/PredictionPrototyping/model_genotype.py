import numpy as np
import numpy.random as rnd
import auto_encoder as adc
import matplotlib.pyplot as plt

# List of parameters needed for one model
class model_genotype_config:

    ## INPUT CHROMOSOME
    input_min_num_ranges = 10
    input_max_num_ranges = 20
    input_min_size_range = 1 # power of two
    input_max_size_range = 2 # power of two

    ## MODEL CHROMOSOME
    min_num_lstm_layers = 2
    max_num_lstm_layers = 4
    min_lstm_layer_size = 2 # power of two
    max_lstm_layer_size = 8 # power of two

    min_num_ffwd_layers = 2
    max_num_ffwd_layers = 4
    min_ffwd_layer_size = 2 # power of two
    max_ffwd_layer_size = 8 # power of two


    ## TARGET CHROMOSOME
    min_target_range = 30   # minutes
    max_target_range = 90   # minutes
    max_lead_time = 60*3    # minutes
    warning_time = 30       # minutes

    ## TRAINING CHROMOSOME
    momentum = 0.9
    max_learning_rate = 0.01
    min_learning_rate = 0.01
    batch_size = 16
    num_batches = 100
    reuse_factor = 10

    ## INCUBATOR SETTINGS
    num_epochs = 50
    mutation_rate = 0.9
    crossover_rate = 0.5
    population_size = 10
    num_generations = 5
    min_cycle_size = 60*6
    folder_name = 'C:/Users/Fredrik/Documents/WashedData/TurbinA'

# Create a genome based on the above specifications
class model_genotype:

    def create_random_multilayer(self,min_layers, max_layers, min_size, max_size):
        depth = rnd.randint(min_layers, max_layers)
        layers = [np.power(2,rnd.randint(min_size, max_size)) for _ in range(depth)]
        return layers
    def mutate_multilayer(self, min_layers, max_layers, min_size, max_size, layers, mutation_rate):
        output = layers
        add_layer = rnd.random() < mutation_rate
        if add_layer:
            if  rnd.random() < 0.5 and len(output) < max_layers:
                output += [np.power(2,rnd.randint(min_size, max_size))]
            elif len(output) > min_layers:
                output = output[:-1]

        should_mutate = rnd.rand(len(output))
        should_mutate = [i < mutation_rate for i in should_mutate]
        for i in range(len(output)):
            if should_mutate[i]:
                if rnd.random() < 0.5 and np.log2(output[i]) < max_size:
                    output[i] = int(output[i] * 2)
                elif np.log2(output[i]) > min_size:
                    output[i] = int(output[i] / 2)
        return output
    def crossover_multilayer(self, min_layers, max_layers, parent0, parent1, crossover_rate):
        output0, output1 = parent0, parent1
        if rnd.random() < crossover_rate:
            try:
                idx11 = rnd.randint(0,len(parent0)-min_layers)
                idx12 = rnd.randint(idx11, idx11+max_layers)
                idx21 = rnd.randint(0,len(parent1)-min_layers)
                idx22 = rnd.randint(idx21, idx21+max_layers)
                segment0 = output0[idx11:idx12]
                segment1 = output1[idx21:idx22]
                output0 = output0[:idx11] + segment1 + output0[idx12:]
                output1 = output1[:idx21] + segment0 + output1[idx22:]
                if not (min_layers < len(output0) <= max_layers):
                    return parent0, parent1
                if not (min_layers < len(output1) <= max_layers):
                    return parent0, parent1
                return output0, output1
            except:
                return parent0, parent1
        return parent0, parent1
    def distance_multilayer(self, layers0, layers1):
        distance = 0
        for i in range(min(len(layers0),len(layers1))):
            distance += abs(layers0[i]-layers1[i])
        if len(layers0) < len(layers1):
            distance += sum(layers1[i+1:])
        elif len(layers0) > len(layers1):
            distance += sum(layers0[i+1:])
        return distance

    def create_random_ranges(self, max_lead_time, min_range, max_range):
        ranges = []
        x = 0
        start_index = 0
        while True:
            end_index = start_index + rnd.randint(min_range,max_range)
            if end_index > max_lead_time:
                break
            ranges += [[start_index, end_index]]
            start_index = end_index
        return ranges
    def mutate_ranges(self, max_lead_time, max_ranges, min_range, max_range, ranges, mutation_rate):
        output = ranges
        if rnd.random() < mutation_rate and len(output) > 2:
           output = output[:-1]

        for i in range(len(output)):
            if rnd.random() > mutation_rate:
                continue
            idx0 = 0
            idx1 = 0
            try:
                idx0 = rnd.randint(output[i-1][1], output[i][1])
            except:
                idx0 = rnd.randint(0, output[i][1])

            try:
                idx1 = rnd.randint(idx0,output[i+1][0])
            except:
                idx1 = rnd.randint(idx0,max_lead_time)
        
            if (min_range < idx1-idx0 < max_range):
                output[i] = [idx0, idx1]

        if rnd.random() < mutation_rate and len(output) < max_ranges:
            idx0 = rnd.randint(output[-1][1], max_lead_time)
            idx1 = rnd.randint(idx0,max_lead_time)

            if min_range < idx1-idx0 < max_range:
                output += [[idx0, idx1]]

        return output
    def crossover_ranges(self, max_ranges, parent0, parent1, crossover_rate):
        if rnd.random() > crossover_rate:
            return parent0, parent1
        try:
            idx11 = rnd.randint(0,len(parent0)-3)
            idx12 = rnd.randint(idx11+2, len(parent0))
            idx21 = False
            idx22 = False
            for i in range(len(parent1)):
                rng = parent1[i]
                if parent0[idx11][1] < rng[0] and not idx21:
                    idx21 = i
                if parent0[idx12][1] < rng[0] and idx21:
                    idx22 = i-1
        except:
            return parent0, parent1

        if idx22-idx21 <= 0:
            return parent0, parent1
        output0, output1 = parent0, parent1
        segment0 = output0[idx11+1:idx12]
        segment1 = output1[idx21:idx22]
        output0 = output0[:idx11] + segment1 + output0[idx12:]
        output1 = output1[:idx21] + segment0 + output1[idx22:]

        if max_ranges < len(output0) or max_ranges < len(output1):
            return parent0, parent1
        return output0, output1
    def distance_ranges(self, ranges0, ranges1):

        return 0 # NOT IMPLEMENTED

    def create_random_window(self, minimum, maximum):

        return rnd.randint(minimum, maximum)

    def spawn_genome(self, config):
        return {'ffwd_layers' : self.create_random_multilayer(config.min_num_lstm_layers,
                                                              config.max_num_lstm_layers,
                                                              config.min_lstm_layer_size,
                                                              config.max_lstm_layer_size),

                'lstm_layers' : self.create_random_multilayer(config.min_num_ffwd_layers, 
                                                              config.max_num_ffwd_layers, 
                                                              config.min_ffwd_layer_size, 
                                                              config.max_ffwd_layer_size),

                'target_ranges' : self.create_random_ranges(config.max_lead_time,  
                                                            config.min_target_range, 
                                                            config.max_target_range),

                'window_size' : self.create_random_window(15,
                                                          config.min_target_range)}

    def mutate_genome(self, config, genome):
        output = genome
        output['ffwd_layers'] = self.mutate_multilayer(config.min_num_lstm_layers, 
                                                  config.max_num_lstm_layers, 
                                                  config.min_lstm_layer_size, 
                                                  config.max_lstm_layer_size,
                                                  genome['ffwd_layers'],
                                                  config.mutation_rate)

        output['lstm_layers'] = self.mutate_multilayer(config.min_num_lstm_layers, 
                                                  config.max_num_lstm_layers, 
                                                  config.min_lstm_layer_size, 
                                                  config.max_lstm_layer_size,
                                                  genome['lstm_layers'],
                                                  config.mutation_rate)

        output['target_ranges'] = self.mutate_ranges(config.max_lead_time, 
                                                config.max_target_ranges, 
                                                config.min_target_range, 
                                                config.max_target_range,
                                                genome['target_ranges'],
                                                config.mutation_rate)
        return output
    def mate_genomes(self, config, parent0, parent1):
        output0, output1 = parent0, parent1
        output0['ffwd_layers'], output1['ffwd_layers'] = self.crossover_multilayer(config.min_num_lstm_layers, 
                                                                              config.max_num_lstm_layers,
                                                                              parent0['ffwd_layers'],
                                                                              parent1['ffwd_layers'],
                                                                              config.crossover_rate)

        output0['lstm_layers'], output1['lstm_layers'] = self.crossover_multilayer(config.min_num_lstm_layers, 
                                                                              config.max_num_lstm_layers,
                                                                              parent0['lstm_layers'],
                                                                              parent1['lstm_layers'],
                                                                              config.crossover_rate)

        output0['target_ranges'], output1['target_ranges'] = self.crossover_ranges(config.max_target_ranges,
                                                                              parent0['target_ranges'],
                                                                              parent1['target_ranges'],
                                                                              config.crossover_rate)
        return output0, output1
    def distance_genomes(self, genome0, genome1):
        distance =  self.distance_multilayer(genome0['ffwd_layers'],genome1['ffwd_layers'])
        distance += self.distance_multilayer(genome0['lstm_layers'],genome1['lstm_layers'])
        return distance

if __name__ == '__main__':
    config = model_genotype_config()
    spawner = model_genotype()
    auto_encoder = adc.auto_encoder('C:/Users/FRCHR/OneDrive - DNV GL/SummerProject2017/Data/System80/WashedData')
    ranges = [spawner.create_random_multilayer(config.input_min_num_ranges, 
                                               config.input_max_num_ranges,
                                               config.input_min_size_range,
                                               config.input_max_size_range) for _ in range(auto_encoder.num_dimensions)]
    scaler = [scale if rnd.random() < 0.5 else 0 for scale in auto_encoder.scaler]
    encoded_time_series = auto_encoder.export_cycles(scaler, ranges, 400)

    plt.figure()
    plt.imshow(encoded_time_series[0], cmap='gray')
    plt.show()
    exit()