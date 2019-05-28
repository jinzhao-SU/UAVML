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
import importlib


def create(cls):
    """expects a string that can be imported as with a module.class name"""
    module_name, class_name = cls.rsplit(".", 1)
    try:
        print('importing ' + module_name)
        somemodule = importlib.import_module(module_name)
        print('getattr ' + class_name)
        cls_instance = getattr(somemodule, class_name)
        print(cls_instance)
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)
    return cls_instance
