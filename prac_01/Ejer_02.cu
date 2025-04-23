#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>

#define N 10000
#define THREADS_PER_BLOCK 256

__global__ void vectorAdd(float *A, float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    std::srand(static_cast<unsigned>(std::time(0)));

    // Vectores en host
    std::vector<float> h_A(N);
    std::vector<float> h_B(N);
    std::vector<float> h_C(N); // resultado

    // Inicializar con valores aleatorios
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Punteros en device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    // Copiar datos al device
    cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Lanzar kernel
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);

    // Copiar resultado al host
    cudaMemcpy(h_C.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Mostrar algunos resultados
    std::cout << "A[0] + B[0] = " << h_A[0] << " + " << h_B[0] << " = " << h_C[0] << std::endl;

    // Liberar memoria en device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
