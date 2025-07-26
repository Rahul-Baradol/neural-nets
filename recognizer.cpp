#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace std;

class Layer {
    int input_size, output_size;

    vector<float> outputs, bias;
    vector<vector<float>> weights;
    
    void init_values(vector<float> &v) {
        int n = v.size();

        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        for (float &ele: v) {
            ele = dist(gen);
        }
    }

    void init_values(vector<vector<float>> &v) {
        int n = v.size(), m = v[0].size();

        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                v[i][j] = dist(gen);
            }
        }
    }

public:
    Layer(int input_size, int output_size) {
        this -> input_size = input_size;
        this -> output_size = output_size;

        outputs.resize(output_size, 0.0);
        bias.resize(output_size, 0.0);
        weights.resize(input_size, vector<float>(output_size, 0));
    }

    void init() {
        init_values(bias);
        init_values(weights);
    }

    vector<vector<float>> get_weights() {
        return weights;
    }

    vector<float> get_bias() {
        return bias;
    }
};

int main() {
    Layer *layer = new Layer(2, 4);

    layer -> init();

    vector<vector<float>> w = layer -> get_weights();

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            cout << w[i][j] << " ";
        }
        cout << "\n";
    }

    delete layer;
    return 0;
}