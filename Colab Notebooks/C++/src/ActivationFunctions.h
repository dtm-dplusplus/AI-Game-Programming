#include <cmath>

// Sign Function
double Sign(double _x) {
    if (_x >= 0.0) return 1.0;
    else return -1.0;
}

// Step Function
double Step(double _x) {
    if (_x >= 0.0) return 1.0;
    else return 0.0;
}

// Sigmoid Function
double Sigmoid(double _x) {
    return 1.0 / (1.0 + std::exp(-_x));
}

// Sigmoid Derivative Function
double SigmoidDelta(double _y) {
    return _y * (1 - _y);
}

// Linear Function
double Linear(double _x) {
    return _x;
}