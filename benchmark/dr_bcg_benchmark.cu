#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

#include <benchmark/benchmark.h>

#include <string>

#include "dr_bcg/dr_bcg.h"
#include "dr_bcg/helper.h"

static bool context_added = false;

class DR_BCG_Benchmark : public benchmark::Fixture {
public:
    DR_BCG_Benchmark() {
        if (!context_added) {
            add_context();
            context_added = true;
        }
    }

private:
    void add_context() {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        benchmark::AddCustomContext("Device", prop.name);
        benchmark::AddCustomContext("Compute Capability", std::to_string(prop.major) + "." + std::to_string(prop.minor));
    }
};

BENCHMARK_DEFINE_F(DR_BCG_Benchmark, BM_DR_BCG)(benchmark::State &state)
{
    const int m = state.range(0);
    const int n = state.range(1);
    constexpr float tolerance = 0.01;
    constexpr int max_iterations = 128;

    std::vector<float> A(m * m);
    fill_spd(A.data(), m);
    std::vector<float> X(m * n, 0);
    std::vector<float> B(m * n);
    fill_random(B.data(), m, n);

    int iterations = 0;

    float *d_A = nullptr;
    float *d_X = nullptr;
    float *d_B = nullptr;

    cublasHandle_t cublasH = NULL;
    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnParams_t cusolverParams = NULL;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUSOLVER_CHECK(cusolverDnCreateParams(&cusolverParams));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * m * m));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_X), sizeof(float) * m * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(float) * m * n));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(float) * m * m, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_X, X.data(), sizeof(float) * m * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), sizeof(float) * m * n, cudaMemcpyHostToDevice));

    // Warmup
    constexpr int warm_up_iterations = 10;
    for (int i = 0; i < warm_up_iterations; i++)
    {
        dr_bcg::dr_bcg(cusolverH, cusolverParams, cublasH, m, n, d_A, d_X, d_B, tolerance, max_iterations, &iterations);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    for (auto _ : state)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        dr_bcg::dr_bcg(cusolverH, cusolverParams, cublasH, m, n, d_A, d_X, d_B, tolerance, max_iterations, &iterations);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_B));

    CUBLAS_CHECK(cublasDestroy_v2(cublasH));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUSOLVER_CHECK(cusolverDnDestroyParams(cusolverParams));
}
BENCHMARK_REGISTER_F(DR_BCG_Benchmark, BM_DR_BCG)->UseManualTime()->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Ranges({{64, 256}, {4, 16}});
