import ANNFunctions as ann

# Threshold for activation functions
THRESHOLD = 0.2 

# Learning rate for weight and error calculation
LEARN_RATE = 0.1

# Epoch Passes (Starts at epoch 1) 
EPOCH_MAX = 20

# Initial input weights
w = [0.3, -0.2]

# Inputs and Desired Outputs
X1 = [0.0, 0.0, 1.0, 1.0]
X2 = [0.0, 1.0, 0.0, 1.0]
YD = [0.0, 0.0, 0.0, 1.0]

# Feed Forward Propogation Algorithm
for epochCount in range(1, EPOCH_MAX- len(X1), 1):
    for i in range(0,4,1):
        # Produce input from random weights
        X = ann.Input(X1[i], w[0], X2[i], w[1])

        # Produce output from input using activation funcntion
        Y = ann.Step(X, THRESHOLD)

        # Calculate error ( error = yD - Y)
        ERROR = ann.Error(YD[i], Y)

        # Adjust weights based on error
        w[0] = ann.Learn(w[0], LEARN_RATE, X1[i], ERROR)
        w[1] = ann.Learn(w[1], LEARN_RATE, X2[i], ERROR)

ann.PrintWeights(w[0], w[1])