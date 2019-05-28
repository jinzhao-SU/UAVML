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

"""Builds model trainer.

Implements the C3D class and the LSTM class.

"""

from base.base_tester import BaseTest


class C3DLSTMTester(BaseTest):
    def __init__(self, model, data, config):
        super(C3DLSTMTester, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks = []

    def test(self):
        scores = self.model.evaluate(self.data[0], self.data[1], batch_size=1, verbose = self.config.tester.verbose_training)
        output = self.model.predict(self.data[0], batch_size=1)
        return scores, output
