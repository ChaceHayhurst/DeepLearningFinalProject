class Adam:
  def __init__(self, learning_rate):
    self.learning_rate = learning_rate
    self.epsilon = 1e-7
    self.beta_1 = 0.9
    self.beta_2 = 0.999
    
    self.m = 0  # First moment zero vector
    self.v = 0  # Second moment zero vector.
    self.t = 0  # Time counter
      
  def call(self, gradients, parameters):
    gradient = gradients[0]
    param = parameters[0]
    self.t += 1

    # TODO: Implement
    self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
    self.v = self.beta_2 * self.v + (1 - self.beta_2) * (gradient * gradient)
    
    m_hat = self.m / (1 - self.beta_1**self.t)
    v_hat = self.v / (1 - self.beta_2**self.t)
    return -(self.learning_rate * m_hat / (v_hat**0.5 + self.epsilon))
