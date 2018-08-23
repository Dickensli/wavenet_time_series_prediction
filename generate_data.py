import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

SAMPLE_RATE_HZ = 200000.0  # Hz
TRAIN_ITERATIONS = 400
SAMPLE_DURATION = 0.5  # Seconds
SAMPLE_PERIOD_SECS = 1.0 / SAMPLE_RATE_HZ
MOMENTUM = 0.95
GENERATE_SAMPLES = 1000
QUANTIZATION_CHANNELS = 256
NUM_SPEAKERS = 3
F1 = 155.56  # E-flat frequency in hz
F2 = 196.00  # G frequency in hz
F3 = 233.08  # B-flat frequency in hz

def make_sine_waves():
    """
    Creates a time-series of sinusoidal audio amplitudes.
    :return: [vm_num,sequence length,feature dim]
    """
    sample_period = 1.0/SAMPLE_RATE_HZ
    times = np.arange(0.0, SAMPLE_DURATION, sample_period)
    amplitudes = 10+(np.sin(times * 2.0 * np.pi * F1) / 3.0 +
                  np.sin(times * 2.0 * np.pi * F2) / 3.0 +
                  np.sin(times * 2.0 * np.pi * F3) / 3.0)

    #------------------------------
    x = [i for i in range(len(times))]
    plt.plot(x, amplitudes)
    plt.show()
    #------------------------------

    amplitudes = np.array(amplitudes)
    amplitudes = np.expand_dims(amplitudes,0)
    amplitudes = np.expand_dims(amplitudes,-1)
    mean_amplitudes = np.zeros((1, times.shape[0], 1))
    amplitudes = np.concatenate([amplitudes,mean_amplitudes],axis=-1)
    print(np.array(amplitudes).shape)


    return amplitudes

def generate_data(output_path):
    """
    store the data
    :param output_path:
    :return: [vm_num,sequence length,feature dim]
    """
    amplitudes = make_sine_waves()
    data= pd.Panel(amplitudes)
    data.to_hdf(output_path,'df')
if __name__=="__main__":
    output_path = "/Users/didi/PycharmProjects/wavenet_clound/data/test_wave.h5"
    generate_data(output_path)