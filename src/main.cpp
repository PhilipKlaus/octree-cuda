#include <iostream>
#include <memory>

#include "tools.cuh"

using namespace std;

constexpr unsigned int CELL_COUNT = 10;

int main() {

    auto data = generate_point_cloud_cuboid(2);
    auto host = data->toHost();

    for(int i = 0; i < 2 * 2 * 2; ++i) {
        cout << "x: " << host[i].x << ", y: " << host[i].y << ", " << host[i].z << endl;
    }
}