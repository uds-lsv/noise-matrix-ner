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

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

class NoiseMatrix:
    
    def __init__(self, name):
        self.name = name
        self.matrix = None
        self.description = None
        self.idx_to_label_name_map = None

    def set_matrix(self, matrix):
        self.matrix = matrix
        
    def set_description(self, description):
        self.description = description
        
    def set_idx_to_label_name_map(self, idx_to_label_name_map):
        """ 
            A map to convert a label index to a specific name
            Used e.g. for the tick-labels of the plot
        """
        self.idx_to_label_name_map = idx_to_label_name_map
                
    @staticmethod
    def load_from_file(name):
        dir_path = "../noise_mats/"
            
        with open(os.path.join(dir_path, "{}.pkl".format(name)), "rb") as input_file:
            return pickle.load(input_file)
    
    def store_to_file(self):
        dir_path = "../noise_mats/"
            
        with open(os.path.join(dir_path, "{}.pkl".format(self.name)), "wb") as output_file:
            pickle.dump(self, output_file)
        
    def visualize(self, title=None, xlabel="noisy label", ylabel="true label", save_filename=None):
        if title is None:
            title = "Noise Matrix {}".format(self.name)
        
        NoiseMatrix.visualize_matrix(self.matrix, title, xlabel, ylabel, self.idx_to_label_name_map, save_filename)
    
    @staticmethod
    def visualize_matrix(matrix, title="", xlabel="noisy label", ylabel="true label", idx_to_label_name_map=None, save_filename=None,
                        vmin=0, vmax=1):
        plt.matshow(matrix, vmin=vmin, vmax=vmax, interpolation="none", cmap=plt.cm.Blues)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.colorbar()
        
        if not idx_to_label_name_map is None:
            tick_marks = np.arange(len(idx_to_label_name_map))
            label_names = [idx_to_label_name_map[idx] for idx in tick_marks]
            plt.xticks(tick_marks, label_names, rotation=90)
            plt.yticks(tick_marks, label_names) 
            plt.title(title,y=1.5)
        else:
            plt.title(title, y=1.2)
        
        if save_filename != None:
            plt.savefig(save_filename, bbox_inches="tight")
            
        return plt.gcf()
            
    @staticmethod
    def compute_noise_matrix(instance_as, instance_bs, num_labels, label_name_to_label_idx_map = None, row_normalize=True):
        """
            For two corresponding lists of clean and noisy instance objects that have a label attribute,
            compute the noise or confusion matrix.
            instance_as: rows in the noise matrix (often clean-data)
            instance_bs: columns in the noise matrix (often noisy-data)
        """
        assert len(instance_as) == len(instance_bs)
        noise_matrix = np.zeros((num_labels, num_labels))

        if label_name_to_label_idx_map is None:
            label_name_to_label_idx_function = lambda l: l # identity function
        else:
            label_name_to_label_idx_function = lambda l: label_name_to_label_idx_map[l]

        for instance_a, instance_b in zip(instance_as, instance_bs):
            label_a = label_name_to_label_idx_function(instance_a.label)
            label_b = label_name_to_label_idx_function(instance_b.label)
            noise_matrix[label_a][label_b] += 1

        if row_normalize:
            for row in noise_matrix:
                row_sum = np.sum(row)
                if row_sum != 0:
                    row /= row_sum
        return noise_matrix
