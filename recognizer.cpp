#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <thread>
#include <chrono>

using namespace std;

class Layer {
    int input_size, output_size;

    vector<float> outputs, bias, weights;
    
    void init_values(vector<float> &v) {
        int n = v.size();

        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        for (float &ele: v) {
            ele = dist(gen);
        }
    }

public:
    Layer(int input_size, int output_size) {
        this -> input_size = input_size;
        this -> output_size = output_size;

        outputs.resize(output_size, 0.0);
        bias.resize(output_size, 0.0);
        weights.resize(input_size * output_size, 0.0);
    }

    void forward(vector<float> &input) {
        if (input.size() != input_size) {
            cerr << "Forward pass failed" << endl;
            cerr << "Invalid input size. Expected " << input_size << " but given " << input.size() << endl;
            return; 
        }

        for (int i = 0; i < output_size; i++) {
            float value = bias[i];
            for (int j = 0; j < input_size; j++) {
                value += input[j] * weights[(i * input_size) + j];
            }

            outputs[i] = value;
        }
    }
 
    void backward(float lr, vector<float> &inputs, vector<float> &outputs, vector<float> &correct_outputs) {
        int w_count = input_size * output_size;

        for (int i = 0; i < w_count; i++) {
            int neuron = i / 100;
            float inp = i % 100;

            float dl_dw = (outputs[neuron] - correct_outputs[neuron]) * inputs[inp];
            weights[i] -= lr * dl_dw;
        }
    }

    void init() {
        init_values(bias);
        init_values(weights);
    }

    vector<float> get_outputs() {
        return outputs;
    }

    vector<float> get_weights() {
        return weights;
    }

    vector<float> get_bias() {
        return bias;
    }
};

class Net {
    Layer *layer;

public:
    Net() {
        layer = new Layer(100, 4);
        layer -> init();
    }

    ~Net() {
        delete layer;
    }

    void forward(vector<float> &input) {
        layer -> forward(input);
    }

    void backward(float lr, vector<float> &inputs, vector<float> &outputs, vector<float> &correct_outputs) {
        layer -> backward(lr, inputs, outputs, correct_outputs);
    }

    vector<float> get_outputs() {
        return layer -> get_outputs();
    }

    vector<float> get_weights() {
        return layer -> get_weights();
    }
    
    vector<float> get_bias() {
        return layer -> get_bias();
    }
};

void train(string file_name, Net *net, vector<float> &correct) {
    fstream file(file_name);
    string line;

    vector<float> input;
    while (getline(file, line)) {
        for (char c: line) {
            if (c == '.') {
                input.push_back(0);
            } else {
                input.push_back(1);
            }
        }
    }
    
    for (int i = 0; i < 1000; i++) {
        net -> forward(input);
        vector<float> output = net -> get_outputs();
    
        net -> backward(0.05, input, output, correct);    
    }
}

vector<float> predict(string file_name, Net *net) {
    fstream file(file_name);
    string line;

    vector<float> input;
    while (getline(file, line)) {
        for (char c: line) {
            if (c == '.') {
                input.push_back(0);
            } else {
                input.push_back(1);
            }
        }
    }

    net -> forward(input);
    return net -> get_outputs();
}

int main() {
    Net *net = new Net();
    
    vector<float> correct2 = {0,0,1,0};
    vector<float> correct3 = {0,0,1,1};

    train("two.txt", net, correct2);
    train("three.txt", net, correct3);
    
    vector<float> ans2 = predict("two.txt", net);
    vector<float> ans3 = predict("three.txt", net);

    for (float ele: ans2) {
        cout << ele << " ";
    }
    cout << "\n\n";

    for (int ele: ans3) {
        cout << ele << " ";
    }
    cout << "\n";

    delete net;
    return 0;
}