#include "SimpleANN.h"

#include <cmath>
#include <vector>
#include <iostream>
#include "ActivationFunctions.h"
#include "ArtificialNeuralNetwork.h"

// Epoch parameters
void SimpleANN()
{
    constexpr int EPOCH_COUNT_MAX = 1000000;
    constexpr double LEARN_RATE = 0.1;
    int epochCount = 0;
    double epochSumError = 0.0;

    // Initialize weights and biases
    double w1to3 = 0.5, w1to4 = 0.9;
    double w2to3 = 0.4, w2to4 = 1.0;
    double w3to5 = -1.2, w4to5 = 1.1;
    double b3 = 0.8, b4 = -0.1, b5 = 0.3;

    // Initialize dataset
    constexpr double x1[]{1.0, 0.0, 1.0, 0.0};
    constexpr double x2[]{1.0, 1.0, 0.0, 0.0};
    constexpr double YD[]{0.0, 1.0, 1.0, 0.0};
    constexpr int INPUT_COUNT_MAX{4};

    double sample[INPUT_COUNT_MAX][2];

    // Multilayer neural network
    while(epochCount < EPOCH_COUNT_MAX){
        epochSumError = 0.0;
        for(int i = 0; i < INPUT_COUNT_MAX; i++) {
            // Forward propagation
            // Neuron 3
            double x3 = b3 + x1[i]*w1to3 + x2[i]*w2to3;
            double y3 = Sigmoid(x3);

            // Neuron 4
            double x4 = b4 + x1[i]*w1to4 + x2[i]*w2to4;
            double y4 = Sigmoid(x4);

            // Neuron 5
            double x5 = b5 + y3*w3to5 + y4*w4to5;
            double y5 = Sigmoid(x5);

            // Back propagation
            // Neuron 5
            double e5 = Error(YD[i], y5);
            double e5Delta = SigmoidDelta(y5) * e5;
            w3to5 += LEARN_RATE * y3 * e5Delta;
            w4to5 += LEARN_RATE * y4 * e5Delta;
            b5 += LEARN_RATE * e5Delta;
            
            // Neuron 3
            double e3Delta = SigmoidDelta(y3) * e5Delta * w3to5;
            w1to3 += LEARN_RATE * x1[i] * e3Delta;
            w2to3 += LEARN_RATE * x2[i] * e3Delta;
            b3 += LEARN_RATE * e3Delta;

            // Neuron 4
            double e4Delta = SigmoidDelta(y4) * e5Delta * w4to5;
            w2to4 += LEARN_RATE * x2[i] * e4Delta;
            w1to4 += LEARN_RATE * x1[i] * e4Delta;
            b4 += LEARN_RATE * e4Delta;

            // Calculate epoch sum error
            double tx3 = x1[i]*w1to3 + x2[i]*w2to3 + b3;
            double ty3 = Sigmoid(tx3);
            double tx4 = x1[i]*w1to4 + x2[i]*w2to4 + b4;
            double ty4 = Sigmoid(tx4);
            double tx5 = ty3*w3to5 + ty4*w4to5 + b5;
            double ty5 = Sigmoid(tx5);
            double te5 = Error(YD[i], ty5);
            epochSumError += pow(e5, 2);

            // Update case results
            sample[i][0] = y5;
            sample[i][1] = e5;
        }
        epochCount++;

        // Repeat training until epoch sum error is less than 0.001
        if(epochSumError < 0.001) break;
    }

    #include <iostream>

    // Print Results
    std::cout << "Epoch Units = " << epochCount << "\n";

    std::cout << "w1to3 = " << w1to3 << "\n";
    std::cout << "w1to4 = " << w1to4 << "\n";
    std::cout << "w2to3 = " << w2to3 << "\n";
    std::cout << "w2to4 = " << w2to4 << "\n";
    std::cout << "w3to5 = " << w3to5 << "\n";
    std::cout << "w4to5 = " << w4to5 << "\n";

    std::cout << "\nb3 = " << b3 << "; b4 = " << b4 << "; b5 = " << b5 << "\n";

    std::cout << "\nEpoch Sum Error = " << epochSumError << "\n";

    for(int i = 0; i < INPUT_COUNT_MAX; i++) {
    std::cout << "X1: " << x1[i] << " X2: " << x2[i] << " YD: " << YD[i] << " Y5: " << sample[i][0] << "\tE: " << sample[i][1] << "\n";
    }
}
    