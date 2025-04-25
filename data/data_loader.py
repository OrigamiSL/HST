import os
import warnings
import copy
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler
from utils.modwt import modwt_v

warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ECL.csv',
                 Batch_size=16, MODWT_level=3, ccc_num=5):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.input_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.input_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.root_path = root_path
        self.data_path = data_path
        self.MODWT_level = MODWT_level  # MODWT_level == Model_hierarchy
        self.ccc_num = ccc_num
        self.eps = 1e-5  # eps
        self.batch_size = Batch_size
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.input_len, len(df_raw) - num_test - self.input_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols = list(df_raw.columns)
        if self.data_path == 'Wind.csv':
            df_raw = df_raw[cols]
        else:
            cols.remove('date')
            df_raw = df_raw[['date'] + cols]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        df_value = df_data.values

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        self.data = self.scaler.transform(df_value)
        self.data_x = self.data[border1:border2]
        self.window_num = self.data_x.shape[0] - self.input_len - self.pred_len + 1
        if self.set_type == 0:
            self.index_list = np.arange(self.window_num)
            self.index_list = np.random.choice(self.index_list, self.window_num, replace=False)

    def __getitem__(self, index):
        if self.set_type == 0:
            seq_x = []
            var_ccc = []
            num_list = np.arange(self.data_x.shape[1])
            if self.data_x.shape[1] > 100:
                num = np.random.randint(low=5 * int(np.log(self.data_x.shape[1])),
                                        high=min(
                                            max(10 * int(np.log(self.data_x.shape[1])), self.data_x.shape[1] // 4),
                                            self.data_x.shape[1]),
                                        size=1)
            else:
                num = self.data_x.shape[1]
            for idx in self.index_list[index * self.batch_size: (index + 1) * self.batch_size]:
                current_index = np.random.choice(num_list, num, replace=False)
                r_begin = idx
                r_end = r_begin + self.input_len + self.pred_len
                seq_x_temp = copy.deepcopy(self.data_x[r_begin:r_end, current_index])
                seq_x.append(seq_x_temp)

                input_x_temp = seq_x_temp[:self.input_len, :]
                vp = modwt_v(input_x_temp, 'db4', self.MODWT_level)  # [MODWT_level + 1, Lin, V]
                vp_mean = np.mean(vp, axis=1, keepdims=True)
                vp_std = np.std(vp, axis=1, keepdims=True)
                normalized_vp = (vp - vp_mean) / vp_std
                vp_ccc = normalized_vp.swapaxes(1, 2) @ normalized_vp  # [MODWT_level + 1, V, V]
                vp_topccc = np.argsort(vp_ccc, axis=-1)[:, :, ::-1][:, :, :self.ccc_num]  # [MODWT_level + 1, V, N]
                var_ccc.append(vp_topccc)

            seq_x = np.stack(seq_x, axis=0)
            var_ccc = np.stack(var_ccc, axis=0)
        else:
            r_begin = index
            r_end = r_begin + self.input_len + self.pred_len
            seq_x = self.data_x[r_begin:r_end]
            input_x = seq_x[:self.input_len, :]
            vp = modwt_v(input_x, 'db4', self.MODWT_level)  # [MODWT_level + 1, Lin, V]
            vp_mean = np.mean(vp, axis=1, keepdims=True)
            vp_std = np.std(vp, axis=1, keepdims=True)
            normalized_vp = (vp - vp_mean) / vp_std
            vp_ccc = normalized_vp.swapaxes(1, 2) @ normalized_vp  # [MODWT_level + 1, V, V]
            var_ccc = np.argsort(vp_ccc, axis=-1)[:, :, ::-1][:, :, :self.ccc_num]  # [MODWT_level + 1, V, N]

        return seq_x, np.ascontiguousarray(var_ccc)

    def __len__(self):
        if self.set_type == 0:
            return (len(self.data_x) - self.input_len - self.pred_len + 1) // self.batch_size
        else:
            return len(self.data_x) - self.input_len - self.pred_len + 1

    def train_shuffle(self):
        if self.set_type == 0:
            self.index_list = np.random.choice(self.index_list, self.window_num, replace=False)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
