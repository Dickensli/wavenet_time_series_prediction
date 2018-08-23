import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import copy
import math
import random
import time
import os


class DataReader(object):

    def __init__(self, data_dir, file_name='data.h5', test_width=2, train_width=2048, num=None, single=False):

        data = pd.read_hdf(os.path.join(data_dir, file_name), 'df')
        np_data = data.values
        self.test_width = test_width
        self.train_width = train_width
        self.single = single
        # one vm or multiple vm
        if num:
            # test data instance: -(2048+288)~:
            self.test_df = np_data[num, -self.test_width - self.train_width:, :]
            self.test_df = np.expand_dims(self.test_df, 0)
            # validation data instance: -(2048+288*2)~-288
            self.val_df = np_data[num, -self.test_width * 2 - self.train_width:-self.test_width, :]
            self.val_df = np.expand_dims(self.val_df, 0)
            # train data instance: :~-288*8
            self.train_df = np_data[num, :-self.test_width * 2, :]
            self.train_df = np.expand_dims(self.train_df, 0)
        else:
            # test data instance: t -(2048+288)~:
            self.test_df = np_data[:, -self.test_width - self.train_width:, :]
            # validation data instance: -(2048+288*2)~-288
            self.val_df = np_data[:, -self.test_width * 2 - self.train_width:-self.test_width, :]
            # train data instance: :~-288*8
            self.train_df = np_data[:, :-self.test_width * 2, :]

        print ('train size', self.train_df.shape)
        print ('val size', self.val_df.shape)
        print ('test size', self.test_df.shape)
        cache_file = "cache"
        cache_dir = os.path.join(data_dir, cache_file)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        save_train_file = os.path.join(cache_dir,
                                       '{}_{}_{}_{}_{}.npy'.format("train", self.train_width, self.test_width, num,
                                                                   file_name))
        save_test_file = os.path.join(cache_dir,
                                      '{}_{}_{}_{}_{}.npy'.format("test", self.train_width, self.test_width, num,
                                                                  file_name))
        save_val_file = os.path.join(cache_dir,
                                     '{}_{}_{}_{}_{}.npy'.format("val", self.train_width, self.test_width, num,
                                                                 file_name))

        if not os.path.exists(save_train_file):
            start = time.time()
            self.train_batch = self.df_reshape(self.train_df)
            self.test_batch = self.df_reshape(self.test_df)
            self.val_batch = self.df_reshape(self.val_df)
            print("rshape time:", time.time() - start)
            np.save(save_train_file, self.train_batch)
            np.save(save_test_file, self.test_batch)
            np.save(save_val_file, self.val_batch)
        else:
            self.train_batch = np.load(save_train_file)
            self.test_batch = np.load(save_test_file)
            self.val_batch = np.load(save_val_file)

        print ('train size', np.array(self.train_batch).shape)
        print ('val size', np.array(self.val_batch).shape)
        print ('test size', np.array(self.test_batch).shape)

    def train_batch_generator(self, batch_size, single=None):
        if single == None:
            single = self.single
        return self.batch_generator(
            batch_size=batch_size,
            df=self.train_batch,
            shuffle=True,
            num_epochs=10000000,
            single=single
        )

    # one time is ok no need to shuffle and epoch
    def val_batch_generator(self, batch_size, single=None):
        if single == None:
            single = self.single
        return self.batch_generator(
            batch_size=batch_size,
            df=self.val_batch,
            shuffle=True,
            num_epochs=1,
            single=single
        )

    # one time is ok no need to shuffle and epoch
    def test_batch_generator(self, batch_size, single=None):
        if single == None:
            single = self.single
        return self.batch_generator(
            batch_size=batch_size,
            df=self.test_batch,
            shuffle=True,
            num_epochs=1,
            single=single
        )

    def get_nonzero(self, data):
        index = 0
        while index < data.shape[0] and data[index, 0] == 0:
            index += 1
        return index

    def df_reshape(self, df, train=True):
        batch_data = []
        batch_label = []
        for i in range(0, df.shape[0]):
            data = df[i]
            index = self.get_nonzero(data)
            index = int(math.floor(index / float(self.test_width))) * self.test_width
            data = data[index:, :]
            # instance (2336,2335,-288)
            for j in range(data.shape[0], self.train_width + self.test_width - 1, -self.test_width):
                # instance (0:2336)
                batch_data.append(data[j - self.test_width - self.train_width:j, :])
        return np.array(batch_data)

    def batch_generator(self, batch_size, df, shuffle=True, num_epochs=100, single=False):
        for i in range(num_epochs):
            if shuffle:
                np.random.shuffle(df)
            for j in range(0, df.shape[0], batch_size):

                if single:
                    dic = {'data': None,
                           'd7mean': None,
                           'label': None
                           }
                    # instance data [a,2048+288-1,:]
                    batch_data = df[j:j + batch_size, :, 0]
                    batch_data = np.expand_dims(batch_data, -1)
                    # mean
                    batch_mean = df[j:j + batch_size, :, 1]
                    batch_mean = np.expand_dims(batch_mean, -1)
                    # instance label [a,-288:,:]
                    batch_label = np.expand_dims(df[j:j + batch_size, -self.test_width:, 0], -1)
                    dic['data'] = batch_data
                    dic['d7mean'] = batch_mean
                    dic['label'] = batch_label
                    yield dic
                else:
                    dic = {'data': None,
                           'd1max': None,
                           'd7max': None,
                           'd7mean': None,
                           'd4mean': None,
                           'minute': None,
                           'week': None,
                           'label': None
                           }
                    # instance data [a,2048+288-1,:]
                    batch_data = df[j:j + batch_size, :-1, 0]
                    batch_data = np.expand_dims(batch_data, -1)
                    # ld max:
                    batch_1d_max = df[j:j + batch_size, :-1, 1]
                    batch_1d_max = np.expand_dims(batch_1d_max, -1)
                    # 7day max
                    batch_7d_max = df[j:j + batch_size, :-1, 2]
                    batch_7d_max = np.expand_dims(batch_7d_max, -1)
                    # 7day mean
                    batch_7d_mean = df[j:j + batch_size, :-1, 3]
                    batch_7d_mean = np.expand_dims(batch_7d_mean, -1)
                    # 4d mean
                    batch_4d_mean = df[j:j + batch_size, :-1, 4]
                    batch_4d_mean = np.expand_dims(batch_4d_mean, -1)
                    # minute
                    batch_minute = df[j:j + batch_size, :-1, 5]
                    batch_minute = np.expand_dims(batch_minute, -1)
                    # day
                    batch_day = df[j:j + batch_size, :-1, 6]
                    batch_day = np.expand_dims(batch_day, -1)

                    # instance label [a,-288:,:]
                    batch_label = np.expand_dims(df[j:j + batch_size, -self.test_width:, 0], -1)
                    dic['data'] = batch_data
                    dic['d1max'] = batch_1d_max
                    dic['d7max'] = batch_7d_max
                    dic['d7mean'] = batch_7d_mean
                    dic['d4mean'] = batch_4d_mean
                    dic['minute'] = batch_minute
                    dic['week'] = batch_day
                    dic['label'] = batch_label
                    yield dic




