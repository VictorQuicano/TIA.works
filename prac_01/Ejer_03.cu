#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

#define M 2  // filas de A y C
#define K 2  // columnas de A y filas de B
#define N 2  // columnas de B y C
#define THREADS_PER_BLOCK 16

__global__ void matrixMulKernel(int* A, int* B, int* C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    std::srand(static_cast<unsigned>(std::time(0)));

    // Host matrices
    std::vector<int> h_A(M * K);
    std::vector<int> h_B(K * N);
    std::vector<int> h_C(M * N);

    // Inicializar matrices con valores aleatorios
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<int>(rand()) % 21;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<int>(rand()) % 21;

    // Device matrices
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copiar datos al device
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Definir bloques e hilos
    dim3 blockDim(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 gridDim((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                 (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    // Ejecutar kernel
    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);

    // Copiar resultado al host
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    if(M < 10 && N < 10 && K < 10) {
        // Mostrar matrices
        std::cout << "Matriz A:" << std::endl;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < K; ++j) {
                std::cout << h_A[i * K + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "Matriz B:" << std::endl;
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < N; ++j) {
                std::cout << h_B[i * N + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "Resultado de la multiplicación de matrices:" << std::endl;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                std::cout << h_C[i * N + j] << " ";
            }
            std::cout << std::endl;
        }
    } else {
        std::cout << "Resultado de la multiplicación de matrices:" << std::endl;
        for (int i = 0; i < M * N; ++i) {
            std::cout << h_C[i] << " ";
        }
        std::cout << std::endl;   
    }

    // Liberar memoria del device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
