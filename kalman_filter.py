import numpy
import random

class KalmanFilterLinear:
  def __init__(self,_A, _B, _H, _x, _P, _Q, _R):
    self.A = _A                      # State transition matrix.
    self.B = _B                      # Control matrix.
    self.H = _H                      # Observation matrix.
    self.current_state_estimate = _x # Initial state estimate.
    self.current_prob_estimate = _P  # Initial covariance estimate.
    self.Q = _Q                       # Estimated error in process.
    self.R = _R                       # Estimated error in measurements.
  def GetCurrentState(self):
    return self.current_state_estimate
  def Step(self,measurement_vector):
    predicted_state_estimate = self.A * self.current_state_estimate
    predicted_prob_estimate = (self.A * self.current_prob_estimate) * self.A.T + self.B*self.Q*self.B.T
    innovation_covariance = self.H*predicted_prob_estimate*self.H.T + self.R

    innovation = measurement_vector - self.H*predicted_state_estimate
    kalman_gain = predicted_prob_estimate * self.H.T * innovation_covariance.getI()
    self.current_state_estimate = predicted_state_estimate + kalman_gain * innovation
    size = self.current_prob_estimate.shape[0]
    self.current_prob_estimate = (numpy.eye(size)-kalman_gain*self.H)*predicted_prob_estimate

class KalmanFilterLinearRandom:
  previous_measureent = numpy.asmatrix(numpy.zeros((2,1)))
  def __init__(self,_A, _B, _H, _x, _P, _Q, _R):
    self.A = _A                      # State transition matrix.
    self.B = _B                      # Control matrix.
    self.H = _H                      # Observation matrix.
    self.current_state_estimate = _x # Initial state estimate.
    self.current_prob_estimate = _P  # Initial covariance estimate.
    self.Q = _Q                      # Estimated error in process.
    self.R = _R                      # Estimated error in measurements.
  def GetCurrentState(self):
    return self.current_state_estimate
  def Step(self,measurement_vector):
    predicted_state_estimate = self.A * self.current_state_estimate
    predicted_prob_estimate = (self.A * self.current_prob_estimate) * self.A.T + self.B*self.Q*self.B.T
    innovation_covariance = self.H*predicted_prob_estimate*self.H.T + self.R
    if measurement_vector.shape != (2, 1) :
        innovation = self.previous_measureent - self.H*predicted_state_estimate
    else:
        innovation = measurement_vector - self.H*predicted_state_estimate
        self.previous_measureent = measurement_vector
    kalman_gain = predicted_prob_estimate * self.H.T * innovation_covariance.getI()
    self.current_state_estimate = predicted_state_estimate + kalman_gain * innovation
    size = self.current_prob_estimate.shape[0]
    self.current_prob_estimate = (numpy.eye(size)-kalman_gain*self.H)*predicted_prob_estimate
