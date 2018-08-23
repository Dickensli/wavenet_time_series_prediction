import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

predict_path = "../data/predictions/predict_out_2018-07-24_13-22.npy"
label_path = "../data/predictions/label_name_2018-07-24_13-22.npy"

predict = np.load(predict_path)
label = np.load(label_path)
print predict.shape
x = [i for i in range(0,predict.shape[1])]
plt.plot(x,predict[0,:,0],'r')
plt.plot(x,label[0,:,0],'b')
plt.show()