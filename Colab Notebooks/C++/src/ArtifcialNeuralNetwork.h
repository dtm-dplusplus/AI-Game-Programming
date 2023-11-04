#include <vector>

// Input Function
double Input(double _x1, double _w1, double _x2, double _w2) {
    return (_x1 * _w1) + (_x2 * _w2);
}

// Error Function
double Error(double _yd, double _y) {
    return _yd - _y;
}

// Learn Function
double Learn(double _w, double _learn_rate, double _x, double _e) {
    return _w + (_learn_rate * _x * _e);
}

// Define sigmoid function
double sigmoid(double _x) {
    return 1.0 / (1.0 + exp(-_x));
}

// Define sigmoid derivative function
double sigmoidDelta(double _y) {
    return _y * (1 - _y);
}