# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Define the model structure.

Implements the C3D class and the LSTM class.

"""

from base.base_data_loader import BaseDataLoader
import numpy as np

class C3DLSTMDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(C3DLSTMDataLoader, self).__init__(config)

        input = np.load(self.config.data_loader.path + self.config.data_loader.input)
        label = np.load(self.config.data_loader.path + self.config.data_loader.label)

        label[label > 0] = 1

        print("Raw Data Size: ")
        print (input.shape)
        print (label.shape)

        slice = int(input.shape[0]*0.85)
        self.X_train = input[:slice]
        self.Y_train = label[:slice]
        self.X_test = input[slice:]
        self.Y_test = label[slice:]

        print("Splitted Data Size: ")
        print (self.X_train.shape)
        print (self.X_test.shape)
        print (self.Y_train.shape)
        print (self.Y_test.shape)

        self.X_train = self.X_train.reshape((-1, 30, 129, 218, 3))
        self.X_test = self.X_test.reshape((-1, 30, 129, 218, 3))

        self.Y_train = self.Y_train.reshape((-1, 30, 28122))
        self.Y_test = self.Y_test.reshape((-1, 30, 28122))

        print("Splitted Data Size: ")
        print (self.X_train.shape)
        print (self.X_test.shape)
        print (self.Y_train.shape)
        print (self.Y_test.shape)

    def get_train_data(self):
        return self.X_train, self.Y_train

    def get_test_data(self):
        return self.X_test, self.Y_test