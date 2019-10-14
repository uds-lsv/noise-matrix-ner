import keras
from keras import backend as K
from keras import initializers
from keras.layers import Dense
from keras.engine.topology import Layer, Input

import numpy as np

import tensorflow as tf


class DynamicTransitionMatrixGeneration(Layer):
    """
    Implementation of the generation of transition matrices 
    based on the features, as proposed by 
    
    Luo et al.: "Learning with Noise: Enhance Distantly Supervised 
    Relation Extractionwith Dynamic Transition Matrix". ACL 2017.
    
    This implements Formula 1 of the paper (what they call
    noise modeling). The output is a matrix T for each instance
    in the batch. T_ij gives the transition matrix of (clean) label 
    i to (noisy) label j.
    
    In Formula 1, the bias b is a single scalar independent of label
    i and label j. As a softmax is applied, this scalar has no effect.
    The author Yansong Feng clarified in a personal conversation
    that this should be b_ij, i.e. there is a scalar bias value for
    each combination of i and j (this is equivalent to other noise 
    matrix approaches like the ones by Bekker and Goldberger or
    Hedderich and Klakow). This implementation uses b_ij instead of
    the b from the paper.
    
    The weight initalization uses the Keras defaults for Dense layers
    as these were unspecified by the paper.
    
    Code partially inspired by keras.core.Dense
    """

    def __init__(self, num_labels, kernel_initializer='glorot_uniform', bias_initializer='zeros', **kwargs):
        self.num_labels = num_labels
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        
        input_dim = input_shape[-1] # size of each feature representation x_n
        
        # for each entry in the transition matrix, a weight vector w_ij exists of
        # the size of the feature_representation x_n
        self.w = self.add_weight(shape=(self.num_labels, self.num_labels, input_dim),
                                      initializer=self.kernel_initializer,
                                      name='w')
        
        # using b_ij instead of b (see comments above)
        self.bias = self.add_weight(shape=(self.num_labels, self.num_labels,),
                                        initializer=self.bias_initializer,
                                        name='bias')
        self.built = True
        
    def call(self, x, mask=None):
        output = K.dot(self.w, K.transpose(x))
        output = K.transpose(output)
        output = K.bias_add(output, self.bias, data_format='channels_last')
        output = K.softmax(output) # row normalize
        return output
    
    def compute_output_shape(self, input_shape):
        # output shape is batch_size * size of transition matrix, i.e.
        # (batch_size, num_labels, self.num_labels)
        return (input_shape[0], self.num_labels, self.num_labels)  
    
    def get_config(self):
        config = {
            'num_labels': self.num_labels,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class TransitionMatrixApplication(Layer):
    """
    Applies the transition matrices on the (clean) label predictions
    to obtain (noisy) label predictions. Formula 2 in the paper, 
    what they call "Transformation".
    
    In contrast to the noise channel by Bekker and Goldberger or
    the noise matrix by Hedderich and Klakow, each instance has 
    its own transition matrix (noise matrix).
    
    Does not apply another softmax on the output, as the input
    is already expected to be a row normalized transition matrix
    and a probability prediction vector.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        # each instance in the batch (i.e. each entry of label_predictions)
        # has its own transition matrix. Multiply them to apply
        # the transition on the label prediction.
        transition_matrices, label_predictions = inputs
        return K.batch_dot(label_predictions, transition_matrices)
    
    def compute_output_shape(self, input_shapes):
        # same as shape of label_predictions
        # (batch_size, num_labels)
        transition_matrices_shape, label_predictions_shape = input_shapes
        return label_predictions_shape
        
def trace_loss(transition_matrices, num_labels, beta):
    """
    Implementation of the generation of transition matrices 
    based on the features, as proposed by 
    
    Luo et al.: "Learning with Noise: Enhance Distantly Supervised 
    Relation Extractionwith Dynamic Transition Matrix". ACL 2017.
    
    This implements Formula 5 or 6, depending on how it is added to 
    the model.
    
    The input is the tensor obtained from the DynamicTransitionMatrixGeneration
    layer, the number of labels (i.e. number of rows or columns of the
    dynamic transition matrix) and the beta scalar that scales this loss.
    
    The negative value of the trace is used (as in the paper). That
    means that a large, positive beta will push the model towards the
    identity matrix, while a negative beta will push the generated
    transition matrices towards the off diagonals (noisy settings).
    """
    eye_tensor = K.eye(num_labels)
    
    def trace_loss_function(y_true,y_pred):
        # Obtaining trace by multiplying with the identity matrix
        # and then summing. This sums up over all identify matrices,
        # but this is fine since the beta factor is the same for 
        # all instances in formula 5. For formula 6, if different 
        # beta factors should be taken into account, different 
        # models with different instanciations of this loss
        # need to be compiled.
        # 
        return beta * -K.sum(transition_matrices * eye_tensor)

    return trace_loss_function

def sum_loss(loss1, loss2):
    """
    Keras only allows multiple losses if multiple outputs exists.
    Since only one output exists (because the trace loss is independent
    of the prediction output as no target exists), this
    loss allows to sum two other losses.
    """
    def sum_loss_function(y_true, y_pred):
        return loss1(y_true, y_pred) + loss2(y_true, y_pred)
    return sum_loss_function

class ClipZeroOneLayer(Layer):
    
    def __init__(self, **kwargs):
        super(ClipZeroOneLayer, self).__init__(**kwargs)
        
    def call(self, x):
        return K.clip(x, 0, 1)

class NoiseMatrixLayer(Dense):
    """
    Implementation of the noise matrix layer (confusion matrix, noise channel)
    used in 
    
    Hedderich & Klakow: 
    Training a Neural Network in a Low-Resource Setting on Automatically
    Annotated Noisy Data, DeepLo 2018
    https://arxiv.org/pdf/1807.00745.pdf
    
    Formulas referenced in the code are from this paper.
    
    Using the concept proposed by 
    
    Goldberger & Ben-Reuven:
    Training deep neural-networks using a noise adaptation layer, ICLR 2017
    https://openreview.net/forum?id=H12GRgcxg
    
    An implementation by Goldberger & Ben-Reuven can be found here:
    https://github.com/udibr/noisy_labels/blob/master/channel.py
    
    The noise layer can be modeled as a dense layer with a softmax activation
    where the input and output size is equal to the number of layers.
    The weight matrix of the dense layer is the noise matrix. The softmax
    ensures that the weight matrix stays a probability matrix.
    
    The weights of the NoiseMatrixLayer can be initalized (see Formula 4)
    using the constructor
    
    channel_weights = np.log(estimated_noise_matrix + 1e-8)
    NoiseMatrixLayer(weights=[channel_weights])    
    """

    def __init__(self, **kwargs):
        kwargs['use_bias'] = False
        kwargs['activation'] = 'softmax'
        units = None # the size of units will be known during build()
        super().__init__(units, **kwargs)

    def build(self, input_shape):
        if len(input_shape) != 2:  # (batch_size, num_labels)
            raise Exception("Input expected to have shape (batch_size, num_labels)")
        
        self.units = input_shape[1]
        super().build(input_shape)

    def call(self, x, mask=None):
        """
        Given the predictions of the base model (estimate of clean labels, vector
        of class probabilities, Formula 1) transforms it to the distribution of 
        noisy labels by multiplying it with the noise matrix (Formula 3)
        """
        # Apply softmax on noise matrix (b_ij) to ensure that still probability matrix
        # Formula 2
        noise_matrix = self.activation(self.kernel)
        
        # The sum over all classes from Formula 3 can be expressed more
        # efficiently as a matrix multiplication. 
        # Multiplying the noise matrix with the vector of label predictions
        # to obtain new vector of label predictions
        return K.dot(x, noise_matrix)
