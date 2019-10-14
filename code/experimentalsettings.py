# Copyright 2019 Saarland University, Spoken Language Systems LSV 
# Author: Lukas Lange, Michael A. Hedderich, Dietrich Klakow
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS*, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
#
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


import json
import os

class ExperimentalSettings:
    """
    Mimics a dictionary to hold the settings of an experiment.
    Loads the settings from a JSON config file. The settings can 
    not be changed to ensure that they are consistent.
    
    Use:
    
    SETTINGS = ExperimentalSettings.load_json("experiment01")
    a = SETTINGS["IMPORTANT_HYPERPARAMETER]
    
    The JSON file must contain one dictionary {...}. The dictionary
    must at least contain the value "NAME" which must be 
    identical to the filename ("NAME.json").
    """
    
    def __init__(self, name):
        """
        name: Name of the file that stores the configuration
                once finalize() is called (.config is added).
        """
        self.name = name
    
    def __getitem__(self, key):
        return self.settings[key]
    
    def __setitem__(self, key, value):
        raise Exception("ExperimentalSettings object can not be changed.")
    
    def __contains__(self, key):
        return key in self.settings
    
    @staticmethod
    def load_json(name, dir_path="../config/"):
        with open(os.path.join(dir_path, name + ".json"), 'r') as f:
            file_content = f.read()
            settings = json.loads(file_content)
            
            if settings["NAME"] != name:
                raise ValueError(f"Name in json is specified as {settings['NAME']}" + 
                                 f"while the name is loaded from a file called {name}")
            
            new_settings_object = ExperimentalSettings(name)
            new_settings_object.settings = settings
            return new_settings_object
 
