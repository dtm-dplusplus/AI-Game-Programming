import numpy as np

# ACTIVATION FUNCTIONS #
# Sign Function
def Sign(X, THRESHOLD = 0.0): 
    if X >= THRESHOLD: return 1.0
    else: return -1.0

# Step function
def Step(X, THRESHOLD = 0.0): 
    if X >= THRESHOLD: return 1.0
    else: return 0.0

# Sigmoid function
def Sigmoid(X): 
   return 1.0 / (1.0 + np.exp(-X))

# Sigmoid Derivative function
def SigmoidDelta(Y): 
   return Y * (1-Y)

def Linear(X, THRESHOLD = 0.0):
    return X - THRESHOLD


