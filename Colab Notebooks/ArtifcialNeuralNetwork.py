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

# Feed Forward algorithm for individual pairs of Input-Outputs
def TrainSingleIOPairs(X1, X2, YD, THRESHOLD, LEARN_RATE, w = [0.3, -0.2], EPOCH_COUNT_MAX = 50):

    # Check Valid IO Data, assert false
    INPUT_COUNT_MAX = len(X1)
    assert(INPUT_COUNT_MAX != len(X2) or INPUT_COUNT_MAX != len(YD), "ERROR- Data set pairs do not match")

    # Epoch Passes (Starts at epoch 1)
    epochCount = 1

    # Feed Forward Propogation Algorithm
    while(epochCount < EPOCH_COUNT_MAX):
        for i in range(0,INPUT_COUNT_MAX,1):
            X = Input(X1[i], w[0], X2[i], w[1])     # Produce input from random weights
            Y = act.Step(X, THRESHOLD)              # Produce output from input using activation funcntion
            E = Error(YD[i], Y)                 # Calculate error ( error = yD - Y)

            # Adjust weights w1 & w2 based on error formula
            w[0] = Learn(w[0], LEARN_RATE, X1[i], E)
            w[1] = Learn(w[1], LEARN_RATE, X2[i], E)
        epochCount += 1

    # Print Results
    print("Epoch Units = " + str(epochCount))
    print("Learning rate = " + str(LEARN_RATE))
    for i in range(0, 2,1): print("w" + str(i) + " = " + str(w[i]))