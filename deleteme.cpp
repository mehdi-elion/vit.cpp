#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <thread>
#include <cinttypes>
#include <algorithm>
#include <iostream>
using namespace std;

// define custom object
struct {
    int n;
    string s;
} myobj;


void print(const std::vector<float>& V){
    for (int i = 0; i < V.size(); i++) {
        std::cout << V[i] << "; ";
    }
}

// main function
int main(int argc, char **argv)
{
    cout << "Hello magggle !!!\n";
    myobj.n = 4;
    myobj.s = "abcd";
    cout << myobj.n + 6 << myobj.s << "\n";
    cout << 0.5f + 5.0f << "\n";
    cout << 1 / 0.5 + 0.5 << "\n";

    const float m[3] = {1.0f, 10.5f, 1.9f};
    const std::vector<float> mv = {1.0f, 10.5f, 1.9f};
    cout << m[1] << "\n";
    cout << "Size of the vector: " << mv.size() << "\n";
    print(mv);

    return 0;
}
