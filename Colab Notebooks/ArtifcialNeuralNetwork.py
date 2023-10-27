import numpy as np
import ActivationFunctions as act

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

def CheckData(X1, X2, YD):
    INPUT_COUNT_MAX = len(X1)
    assert(INPUT_COUNT_MAX != len(X2) or INPUT_COUNT_MAX != len(YD), "ERROR- Data set pairs do not match")
    return INPUT_COUNT_MAX
