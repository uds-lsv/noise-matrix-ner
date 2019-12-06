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


import copy
import logging

from collections import Counter

import numpy as np
import fasttext

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import conlleval

class Instance:
    
    def __init__(self, word, label, left_context, right_context):
        self.word = word
        self.label = label
        self.left_context = left_context
        self.right_context = right_context
        self.clusterID = -1
        
    def __str__(self):
        return "{} ({}) [{},{},c{}]".format(self.word, self.label, self.left_context, self.right_context, self.clusterID)
    
class DataCreation:
    
    def __init__(self, input_separator=" ", padding_word_string="<none>"):
        self.input_separator = input_separator
        self.padding_word_string = padding_word_string
        self.word_cluster = None
    
    def remove_label_prefix(self, label):
        """ CoNLL2003 distinguishes between I- and B- labels,
            e.g. I-LOC and B-LOC. Drop this distinction to
            reduce the number of labels/increase the number
            of instances per label.
        """
        if label.startswith("I-") or label.startswith("B-"):
            return label[2:]
        else:
            return label
        
    def pad_before(self, a_list, target_length):
        return (target_length - len(a_list)) * [self.padding_word_string] + a_list

    def pad_after(self, a_list, target_length):
        return a_list + (target_length - len(a_list)) * [self.padding_word_string]
    
    def load_connl_dataset(self, path, context_length, remove_label_prefix=False):
        with open(path, mode="r", encoding="utf-8") as input_file:
            lines = input_file.readlines()
        
        tokens = []
        labels = []
        for line in lines:
            line = line.strip()
        
            # skip begin of document indicator (used in some files)
            if line.startswith("-DOCSTART-"):
                continue
            
            # skip empty line / end of sentence marker (used in some files) 
            if len(line) == 0:
                continue
            
            # Skip marker (used in some files)
            if line == "--": 
                continue
                
            elements = line.split(self.input_separator)
            
            tokens.append(elements[0])
            # Take last element of this line as label, in between there might be e.g. the POS tag which we ignore here
            if len(elements) > 1: 
                labels.append(elements[-1])
            else:
                raise Exception(f"Line {line} did not provide a label. Elements are {elements}")
        
        assert len(tokens) == len(labels)
        
        instances = []
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if remove_label_prefix:
                label = self.remove_label_prefix(label)

            left_context = tokens[max(0,i-context_length):i]
            right_context = tokens[i+1:i+1+context_length]

            left_context = self.pad_before(left_context, context_length)
            right_context = self.pad_after(right_context, context_length)
            instances.append(Instance(token, label, left_context, right_context))
        
        return instances
    
class WordCluster:
    
    def __init__(self):
        self.num_clusters = 0
        self.word_cluster = {} 
        
    def load_kmeans_cluster(self, instances, embedded_instances, num_clusters, num_pca_components=100, seed=0):
        '''Applies PCA to embedded values and performs k-Means clustering on training texts
        instances -- List[Instance] is a list of words from the training data
        embedded_instances -- list of embedded vectors of objects from "instances"
        num_cluster -- number of clusters used for kMeans clustering
        num_pca_components -- number of PCA components used for dimensionality reduction
        seed -- random seed for clustering
        '''
        logging.debug(f"Apply PCA to embedded values. Reducing to {num_pca_components} components.")
        self.num_clusters = num_clusters
        
        cluster_values = embedded_instances
        pca = PCA(n_components=num_pca_components)
        pca_results = pca.fit_transform(cluster_values)
        
        # remove duplicate items and give an appropriate weight to single items
        c = Counter([token.word for token in instances])
        vectors, weights, seen, pca_vectors = [], [], [], {}
        for i, token in enumerate(instances):
            item = token.word
            if item not in seen:
                pca_vectors[item] = pca_results[i]
                vectors.append(pca_results[i])
                weights.append(c[item])
                seen.append(item)
        logging.debug(f"Apply kMeans to create {num_clusters} cluster.")
        kmeans = KMeans(n_clusters=num_clusters, random_state=seed).fit(vectors, weights)
        cluster_labels = kmeans.predict(pca_results)
        self.word_cluster = {token.word: cluster_labels[i] for i, token in enumerate(instances)}
    
    def load_brown_cluster(self, brown_cluster_file):
        logging.debug(f"Loading Word Cluster from {brown_cluster_file}")
            
        words, cluster_names = {}, set()  # maps words to clusters
        with open(brown_cluster_file, mode="r", encoding="utf-8") as input_file:
            for line in input_file:
                cluster, word, count = line.split('\t')
                words[word] = int(cluster, 2)
                cluster_names.add(int(cluster, 2))
        self.num_clusters = len(cluster_names)
        
        # as names are in tree form, we rename the cluster to numbers in [0, n[
        new_cluster_names = {}
        for i in range(len(cluster_names)):
            if i not in cluster_names:
                max_name = max(cluster_names)
                new_cluster_names[max_name] = i
                cluster_names.remove(max_name)
            else:
                new_cluster_names[i] = i
        self.word_cluster = {w:new_cluster_names[c] for w,c in words.items()}
    
    def get_cluster(self, instances):
        '''Populates the clusterID field of each token. 
        Will assign -1 if the token is not part of any cluster or no cluster was initialized.
        instances -- List[Instance] is a list of of tokens
        '''
        for token in instances:
            if token.word in self.word_cluster:
                token.clusterID = self.word_cluster[token.word]
            else:
                token.clusterID = -1
        
    
class WordEmbedding:
    
    def __init__(self, padding_word_string="<none>", padding_word_vector=np.zeros(300), unknown_word_vector=np.zeros(300)):
        self.padding_word_string = padding_word_string
        self.padding_word_vector = padding_word_vector
        self.unknown_word_vector = unknown_word_vector
    
    def load_fasttext(self, fasttext_path):
        logging.debug(f"Loading FastText from {fasttext_path}")

        embedding = fasttext.load_model(fasttext_path)
        self.embedding_model = FastTextWrapper(embedding)
        self.embedding_vector_size = 300
        
    def word_to_embedding(self, word):
        if word == self.padding_word_string:
            return self.padding_word_vector
        elif word in self.embedding_model:
            return self.embedding_model[word]
        else:
            return self.unknown_word_vector
    
    def embed_instance(self, instance):
        instance.word_emb = self.word_to_embedding(instance.word)
        instance.left_context_emb = [self.word_to_embedding(word) for word in instance.left_context]
        instance.right_context_emb = [self.word_to_embedding(word) for word in instance.right_context]
        
    def embed_instances(self, instances):
        for instance in instances:
            self.embed_instance(instance)
       
    def instances_to_vectors(self, instances):
        xs = []
        for instance in instances:
            x = []
            x.extend(instance.left_context_emb)
            x.append(instance.word_emb)
            x.extend(instance.right_context_emb)
            xs.append(x)
        xs = np.asarray(xs)
        return xs

class FastTextWrapper:
    """ Wraps FastText to behave like a Gensim model,
        i.e. to allow access like model["word"]
    """
    
    def __init__(self, fasttext_model):
        self.fasttext_model = fasttext_model
    
    def __getitem__(self, word):
        return self.fasttext_model.get_word_vector(word)
    
    def __contains__(self, word):
        # FastText can create embeddings for all words. So
        # there are no OOVs for which the zero embedding
        # needs to be used (For the OOVs in the sense of
        # words that were not seen during training, one
        # needs to check it differently. But this is not
        # needed here).
        return True
    
class LabelRepresentation:
    
    def use_specific_label_map(self, label_name_to_label_idx_map):
        self.label_name_to_label_idx_map = label_name_to_label_idx_map
        self._compute_label_idx_to_label_name_map()
    
    def use_connl_io_labels(self):
        self.use_connl_normalized_labels()
    
    def use_connl_normalized_labels(self):
        self.use_specific_label_map({"O": 0, "PER": 1, "ORG": 2, "LOC": 3, "MISC": 4})
    
    def use_conll_iob_labels(self):
        self.use_conll_bio_labels()
    
    def use_connl_bio_labels(self):
        self.use_specific_label_map({"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "B-MISC": 4, 
                                     "I-ORG": 5, "B-LOC": 6, "I-LOC": 7, "I-MISC": 8})    
       
    def _compute_label_idx_to_label_name_map(self):
        self.label_idx_to_label_name_map = {v:k for k,v in self.label_name_to_label_idx_map.items()}
    
    def get_num_labels(self):
        return len(self.label_name_to_label_idx_map)
    
    def label_name_to_label_idx(self, label):
        return self.label_name_to_label_idx_map[label]
    
    def label_idx_to_label_name(self, label_idx):
        return self.label_idx_to_label_name_map[label_idx]
    
    def label_name_to_one_hot_vector(self, label):
        vector = np.zeros(self.get_num_labels())
        vector[self.label_name_to_label_idx_map[label]] = 1
        return vector
    
    def embed_instances(self, instances):
        for instance in instances:
            instance.label_emb = self.label_name_to_one_hot_vector(instance.label)
            
    def instances_to_vectors(self, instances):
        ys = []
        for instance in instances:
            ys.append(instance.label_emb)
        ys = np.asarray(ys)
        return ys
            
    @staticmethod
    def predictions_to_one_hot(predictions):
        labels = np.argmax(predictions, axis=-1)
        return np.eye(len(predictions[0]))[labels]
    
    def predictions_to_labels(self, predictions):
        label_idxes = np.argmax(predictions, axis=-1)
        return [self.label_idx_to_label_name(label_idx) for label_idx in label_idxes]
    
    @staticmethod
    def convert_io_to_bio_labels(old_labels):
        """ Converts a list of IO labels (e.g. ["O", "ORG", "PER", "PER"])
            to BIO labels (["O", "B-ORG", "B-PER", "I-PER"]). IO labels contain
            less information than BIO labels, so adjacent entities might
            be joined (which is however very rare in practice).
        """
        outside_token = "O"

        new_labels = []
        for i, label in enumerate(old_labels):
            if label == outside_token:
                new_labels.append(outside_token)
            else:
                if i > 0 and old_labels[i-1] == label:
                    new_labels.append("I-" + label)
                else:
                    new_labels.append("B-" + label)
        return new_labels
    
class Evaluation:
    
    def __init__(self, separator=" "):
        self.separator = separator
    
    def create_connl_evaluation_format(self, instances, prediction_labels):
        # CoNLL evaluation script (in perl) expects the format "word true_label predicted_label"
        assert len(instances) == len(prediction_labels)

        output = ""
        for instance, prediction_label in zip(instances, prediction_labels):
            output += "{}{}{}{}{}\n".format(instance.word, self.separator, instance.label, self.separator, prediction_label)
        return output
    
    def evaluate_evaluation_string(self, connl_evaluation_string):            
        counts = conlleval.evaluate(connl_evaluation_string.split('\n'), {'delimiter': self.separator})
        return conlleval.report(counts)
    
    @staticmethod
    def extract_f_score(evaluation_output):
        """ Extracts from the output given by the CoNLL Perl script
            the value corresponding to the total F1 score.
        """
        line = evaluation_output.split("\n")[1]
        return float(line[-5:])
    
    def simple_evaluate(self, instances, prediction_labels):
        """ Returns just the f-score (for all NER types)
            Predictions is a label ("MISC", "ORG", etc. not a class vector!)
        """
        connl_evaluation_string = self.create_connl_evaluation_format(instances, prediction_labels)
        evaluation_output = self.evaluate_with_perl_script(connl_evaluation_string)
        return Evaluation.extract_f_score(evaluation_output)
