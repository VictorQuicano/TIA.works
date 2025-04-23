#include <iostream>
#include <cuda_runtime.h>

using namespace std;

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        cerr << "Error al obtener el número de dispositivos: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    cout << "Número de dispositivos CUDA: " << deviceCount << endl;

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        cout << "\n=== Dispositivo " << i << " ===" << endl;
        cout << "Nombre: " << prop.name << endl;
        cout << "Capacidad de cómputo: " << prop.major << "." << prop.minor << endl;
        cout << "Memoria global total: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << endl;
        cout << "Número de multiprocesadores: " << prop.multiProcessorCount << endl;
        cout << "Tamaño de warp: " << prop.warpSize << endl;
        cout << "Memoria compartida por bloque: " << prop.sharedMemPerBlock << " bytes" << endl;
        cout << "Hilos máximos por bloque: " << prop.maxThreadsPerBlock << endl;
        cout << "Tamaño máximo de bloques (x,y,z): ("
                  << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", "
                  << prop.maxThreadsDim[2] << ")" << endl;
        cout << "Tamaño máximo de grid (x,y,z): ("
                  << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", "
                  << prop.maxGridSize[2] << ")" << endl;
    }

    return 0;
}
