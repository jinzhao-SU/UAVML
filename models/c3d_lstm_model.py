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
import tensorflow as tf
from base.base_model import BaseModel

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation, LSTM, Reshape
from tensorflow.keras.layers import TimeDistributed

class C3DLSTMModel(BaseModel):
    def __init__(self, config):
        super(C3DLSTMModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        cnn_model = Sequential()
        cnn_model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=(129, 218, 3)))
        cnn_model.add(BatchNormalization())
        cnn_model.add(LeakyReLU(alpha=.001))
        cnn_model.add(Conv2D(32, (3, 3)))
        cnn_model.add(BatchNormalization())
        cnn_model.add(LeakyReLU(alpha=.001))
        cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
        cnn_model.add(Dropout(0.25))

        cnn_model.add(Conv2D(64, (3, 3), padding='same'))
        cnn_model.add(BatchNormalization())
        cnn_model.add(LeakyReLU(alpha=.001))
        cnn_model.add(Conv2D(64, (3, 3)))
        cnn_model.add(BatchNormalization())
        cnn_model.add(LeakyReLU(alpha=.001))
        cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
        cnn_model.add(Dropout(0.25))

        cnn_model.add(Flatten())
        cnn_model.add(Dense(512))
        cnn_model.add(LeakyReLU(alpha=.001))
        cnn_model.add(Dropout(0.25))

        lstm_model = Sequential()
        lstm_model.add(Reshape((30, 512), input_shape=(30, 512,)))
        lstm_model.add(LSTM(1024, batch_input_shape=(1, 512), dropout=0.15, return_sequences=True))
        lstm_model.add(BatchNormalization())
        lstm_model.add(LSTM(512, dropout=0.15, return_sequences=False))
        lstm_model.add(Dense(1024))
        lstm_model.add(Dense(28122))
        # lstm_model.add(Activation('sigmoid'))

        series_input = Input(shape=(30, 129, 218, 3))
        encoded_series_input = TimeDistributed(cnn_model)(series_input)
        series_output = lstm_model(encoded_series_input)

        cnn_lstm_model = Model(inputs = series_input, outputs = series_output)

        self.model = cnn_lstm_model



