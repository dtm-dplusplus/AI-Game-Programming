import numpy as np

# ACTIVATION FUNCTIONS #
# Sign Function
def Sign(X, THRESHOLD): 
    if X >= THRESHOLD: return 1
    else: return -1

# Step function
def Step(X, THRESHOLD): 
    if X >= THRESHOLD: return 1
    else: return 0

# Sigmoid function
def Sigmoid(X, THRESHOLD): 
   return (1 / (1 + np.exp(-(X- THRESHOLD))))


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


# LOGGING #
def PrintEpochs(EPOCHS_COUNT):
    print("Epoch Units = " + str(EPOCHS_COUNT))

def PrintTest(test):
    print(test)

def PrintWeights(WEIGHT):
    i = 0
    while(i < len(WEIGHT)):
        print("w" + str(i) + " = " + str(WEIGHT[i]))
        i += 1


