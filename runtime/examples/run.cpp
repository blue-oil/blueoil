/* Copyright 2018 The Blueoil Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "blueoil.hpp"

#include <string>
#include <iostream>
#include <vector>


using namespace std;

// TODO: delete this func. it is for debug.
blueoil::Tensor RandomImage(int height, int width, int channel) {
    blueoil::Tensor t({height, width, channel});
    vector<float>& data = t.data();

    for (int i = 0; i < data.size(); ++i) {
        const float f_rand = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 255;
        data[i] = f_rand;
    }

    return t;
}



int main() {
    string meta_yaml = "meta.yaml";

    blueoil::Predictor predictor = blueoil::Predictor(meta_yaml);

    cout << "classes: " << endl;
    for (string j: predictor.classes) {
        std::cout << j << "\n";
    }

    cout << "task: " << predictor.task << endl;

    cout << "expected input shape: " << endl;
    for (int j: predictor.expected_input_shape) {
        std::cout << j << "\n";
    }

    blueoil::Tensor random_image = RandomImage(256, 256, 3);

    cout << "Run" << endl;
    blueoil::Tensor output =  predictor.Run(random_image);

    cout << "Results !" << endl;
    for (float j: output.data()) {
        cout << j << endl;
    }



}
