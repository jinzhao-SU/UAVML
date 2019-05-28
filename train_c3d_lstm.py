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

"""Train the models

Implements the C3D class and the LSTM class.
inference_c3d(): Builds the models as far as is required for running the network
forward to make predictions.
"""
from data_loader.c3d_lstm_loader import C3DLSTMDataLoader
from models.c3d_lstm_model import C3DLSTMModel
from trainers.c3d_lstm_trainer import C3DLSTMTrainer
from testers.c3d_lstm_tester import C3DLSTMTester
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print("Create the data generator.")
    data_loader = C3DLSTMDataLoader(config)

    print("Create the C3D LSTM model.")
    c3d_model = C3DLSTMModel(config)

    print('Create the trainer')
    trainer = C3DLSTMTrainer(c3d_model.model, data_loader.get_train_data(), config)

    print('Start training the model.')
    trainer.train()

    print('Create tester model')
    test_model = trainer.model

    print('Create the tester')
    tester = C3DLSTMTester(test_model, data_loader.get_test_data(), config)

    print('Start testing the model.')
    scores, output = tester.test()

    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    print(output.shape)


if __name__ == '__main__':
    main()