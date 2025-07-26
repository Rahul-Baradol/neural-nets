#include <iostream>
#include <vector>
#include <cmath>
#include <random>

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
                value += input[j] * weights[(i * output_size) + j];
            }

            outputs[i] = value;
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
        layer = new Layer(2, 4);
        layer -> init();
    }

    ~Net() {
        delete layer;
    }

    void forward(vector<float> &input) {
        layer -> forward(input);
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

int main() {
    Net *net = new Net();

    vector<float> outputs = net -> get_outputs();
    for (float ele: outputs) {
        cout << ele << " ";
    }
    cout << "\n";

    net -> forward(something);

    outputs = net -> get_outputs();
    for (float ele: outputs) {
        cout << ele << " ";
    }
    cout << "\n";


    delete net;
    return 0;
}