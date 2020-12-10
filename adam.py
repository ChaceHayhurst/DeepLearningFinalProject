from collections import defaultdict
import tensorflow as tf
#version 12/9
class Adam:
  def __init__(self, learning_rate):
    self.learning_rate = learning_rate
    self.epsilon = 1e-7
    self.beta_1 = 0.9
    self.beta_2 = 0.999
    
    self.m = None  # First moment vectors
    self.v = None  # Second moment vectors.
    self.t = 0  # Time counter
      
  def call(self, gradients, parameters):
    if self.m is None: 
        self.m = [tf.zeros(t.shape) for t in gradients] 
    if self.v is None: 
        self.v = [tf.zeros(t.shape) for t in gradients]
    self.t += 1.0
    
    change = []
    
    for i, gradient in enumerate(gradients): 
        self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * gradient
        self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * (gradient * gradient)
        
        m_hat = self.m[i] / (1 - self.beta_1**self.t)
        v_hat = self.v[i] / (1 - self.beta_2**self.t)
    
        change += [-(self.learning_rate * m_hat / (v_hat**0.5 + self.epsilon))]

    return change
