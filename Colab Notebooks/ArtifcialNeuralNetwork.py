import numpy as np

# EPOCH FUNCTIONS #
# Input Function
def Input(X1,W1, X2,W2):
    return ((X1*W1) + (X2*W2))

# Error Function
def Error(YD, Y):
    return YD - Y

# Input Function
def Learn(W, LEARN_RATE, X, E):
    return (W + (LEARN_RATE * X * E))