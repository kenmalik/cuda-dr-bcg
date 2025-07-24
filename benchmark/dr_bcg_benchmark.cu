#include <functional>

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

#include <benchmark/benchmark.h>

#include <string>

#include "dr_bcg/dr_bcg.h"
#include "dr_bcg/helper.h"
#include "dr_bcg/device_buffer.h"

#define TIME_CUDA(function)                     \
    do                                          \
    {                                           \
        cudaEvent_t start, stop;                \
        cudaEventCreate(&start);                \
        cudaEventCreate(&stop);                 \
                                                \
        cudaEventRecord(start);                 \
        function;                               \
        cudaEventRecord(stop);                  \
        cudaEventSynchronize(stop);             \
                                                \
        float ms = 0;                           \
        cudaEventElapsedTime(&ms, start, stop); \
        state.SetIterationTime(ms / 1000.0);    \
    } while (0);

static bool context_added = false;

class CUDA_Benchmark : public benchmark::Fixture
{
public:
    CUDA_Benchmark()
    {
        if (!context_added)
        {
            add_context();
            context_added = true;
        }
    }

private:
    void add_context()
    {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        benchmark::AddCustomContext("device", prop.name);
        benchmark::AddCustomContext("compute_capability", std::to_string(prop.major) + "." + std::to_string(prop.minor));
    }
};

class DR_BCG_Benchmark : public CUDA_Benchmark
{
protected:
    cublasHandle_t cublasH = NULL;
    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnParams_t cusolverParams = NULL;

    DR_BCG_Benchmark()
    {
        CUBLAS_CHECK(cublasCreate(&cublasH));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
        CUSOLVER_CHECK(cusolverDnCreateParams(&cusolverParams));
    }

    ~DR_BCG_Benchmark()
    {
        cublasDestroy_v2(cublasH);
        cusolverDnDestroy(cusolverH);
        cusolverDnDestroyParams(cusolverParams);
    }

    std::tuple<float *, float *, float *> initialize_inputs(const int m, const int n)
    {
        float *d_A = nullptr;
        float *d_X = nullptr;
        float *d_B = nullptr;

        std::vector<float> A(m * m);
        fill_spd(A.data(), m);
        std::vector<float> X(m * n, 0);
        std::vector<float> B(m * n);
        fill_random(B.data(), m, n);

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * m * m));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_X), sizeof(float) * m * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(float) * m * n));

        CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(float) * m * m, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_X, X.data(), sizeof(float) * m * n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B.data(), sizeof(float) * m * n, cudaMemcpyHostToDevice));

        return {d_A, d_X, d_B};
    }

    DeviceBuffer filled_device_buffer(
        cusolverDnHandle_t &cusolverH, cusolverDnParams_t &cusolverParams, cublasHandle_t &cublasH,
        const int m, const int n,
        float *d_A, float *d_X, float *d_B)
    {
        DeviceBuffer d(m, n);

        float *d_R;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_R), sizeof(float) * m * n));

        // R = B - AX
        dr_bcg::get_R(cublasH, d_R, m, n, d_A, d_X, d_B);
        dr_bcg::thin_qr(cusolverH, cusolverParams, cublasH, d.w, d.sigma, m, n, d_R);
        CUDA_CHECK(cudaFree(d_R)); // Never used later

        // s = w
        CUDA_CHECK(cudaMemcpy(d.s, d.w, sizeof(float) * m * n, cudaMemcpyDeviceToDevice));

        return d;
    }
};

class QR_Benchmark : public CUDA_Benchmark
{
};

BENCHMARK_DEFINE_F(DR_BCG_Benchmark, DR_BCG)(benchmark::State &state)
{
    constexpr float tolerance = 1e-6;
    constexpr int max_iterations = 2048;

    int iterations = 0;

    const int m = state.range(0);
    const int n = state.range(1);
    auto [d_A, d_X, d_B] = initialize_inputs(m, n);

    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto _ : state)
    {
        TIME_CUDA(dr_bcg::dr_bcg(cusolverH, cusolverParams, cublasH, m, n, d_A, d_X, d_B, tolerance, max_iterations, &iterations));
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_B));

    state.counters["performed_algorithm_iterations"] = iterations;
    state.counters["max_algorithm_iterations"] = max_iterations;
}
BENCHMARK_REGISTER_F(DR_BCG_Benchmark, DR_BCG)
    ->MinWarmUpTime(1.0)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Ranges({{2048, 8192}, {4, 16}});

BENCHMARK_DEFINE_F(DR_BCG_Benchmark, get_xi)(benchmark::State &state)
{
    const int m = state.range(0);
    const int n = state.range(1);
    auto [d_A, d_X, d_B] = initialize_inputs(m, n);

    DeviceBuffer d = filled_device_buffer(cusolverH, cusolverParams, cublasH, m, n, d_A, d_X, d_B);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto _ : state)
    {
        TIME_CUDA(dr_bcg::get_xi(cusolverH, cusolverParams, cublasH, m, n, d, d_A));
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_B));
}
BENCHMARK_REGISTER_F(DR_BCG_Benchmark, get_xi)
    ->MinWarmUpTime(1.0)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Ranges({{2048, 8192}, {4, 16}});

BENCHMARK_DEFINE_F(DR_BCG_Benchmark, get_next_X)(benchmark::State &state)
{
    const int m = state.range(0);
    const int n = state.range(1);
    auto [d_A, d_X, d_B] = initialize_inputs(m, n);

    DeviceBuffer d = filled_device_buffer(cusolverH, cusolverParams, cublasH, m, n, d_A, d_X, d_B);
    CUDA_CHECK(cudaDeviceSynchronize());

    dr_bcg::get_xi(cusolverH, cusolverParams, cublasH, m, n, d, d_A);
    
    // Keep copy of X for consistent benchmark state
    std::vector<float> h_X(m * n);
    CUDA_CHECK(cudaMemcpy(h_X.data(), d_X, sizeof(float) * h_X.size(), cudaMemcpyDeviceToHost));

    for (auto _ : state)
    {
        CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), sizeof(float) * h_X.size(), cudaMemcpyHostToDevice));
        TIME_CUDA(dr_bcg::get_next_X(cublasH, m, n, d.s, d.xi, d.temp, d.sigma, d_X));
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_B));
}
BENCHMARK_REGISTER_F(DR_BCG_Benchmark, get_next_X)
    ->MinWarmUpTime(1.0)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Ranges({{2048, 8192}, {4, 16}});

BENCHMARK_DEFINE_F(DR_BCG_Benchmark, get_w_zeta)(benchmark::State &state)
{
    const int m = state.range(0);
    const int n = state.range(1);
    auto [d_A, d_X, d_B] = initialize_inputs(m, n);

    DeviceBuffer d = filled_device_buffer(cusolverH, cusolverParams, cublasH, m, n, d_A, d_X, d_B);
    CUDA_CHECK(cudaDeviceSynchronize());

    dr_bcg::get_xi(cusolverH, cusolverParams, cublasH, m, n, d, d_A);
    dr_bcg::get_next_X(cublasH, m, n, d.s, d.xi, d.temp, d.sigma, d_X);

    // Keep copies of w and zeta for consistent benchmark state
    std::vector<float> h_w(m * n);
    std::vector<float> h_zeta(n * n);
    CUDA_CHECK(cudaMemcpy(h_w.data(), d.w, sizeof(float) * h_w.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_zeta.data(), d.zeta, sizeof(float) * h_zeta.size(), cudaMemcpyDeviceToHost));

    for (auto _ : state)
    {
        CUDA_CHECK(cudaMemcpy(d.w, h_w.data(), sizeof(float) * h_w.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.zeta, h_zeta.data(), sizeof(float) * h_zeta.size(), cudaMemcpyHostToDevice));
        TIME_CUDA(dr_bcg::get_w_zeta(cusolverH, cusolverParams, cublasH, m, n, d, d_A));
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_B));
}
BENCHMARK_REGISTER_F(DR_BCG_Benchmark, get_w_zeta)
    ->MinWarmUpTime(1.0)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Ranges({{2048, 8192}, {4, 16}});

BENCHMARK_DEFINE_F(DR_BCG_Benchmark, get_s)(benchmark::State &state)
{
    const int m = state.range(0);
    const int n = state.range(1);
    auto [d_A, d_X, d_B] = initialize_inputs(m, n);

    DeviceBuffer d = filled_device_buffer(cusolverH, cusolverParams, cublasH, m, n, d_A, d_X, d_B);
    CUDA_CHECK(cudaDeviceSynchronize());

    dr_bcg::get_xi(cusolverH, cusolverParams, cublasH, m, n, d, d_A);
    dr_bcg::get_next_X(cublasH, m, n, d.s, d.xi, d.temp, d.sigma, d_X);
    dr_bcg::get_w_zeta(cusolverH, cusolverParams, cublasH, m, n, d, d_A);

    // Keep copy of s for consistent benchmark state
    std::vector<float> h_s(m * n);
    CUDA_CHECK(cudaMemcpy(h_s.data(), d.s, sizeof(float) * h_s.size(), cudaMemcpyDeviceToHost));

    for (auto _ : state)
    {
        CUDA_CHECK(cudaMemcpy(d.s, h_s.data(), sizeof(float) * h_s.size(), cudaMemcpyHostToDevice));
        TIME_CUDA(dr_bcg::get_s(cublasH, m, n, d));
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_B));
}
BENCHMARK_REGISTER_F(DR_BCG_Benchmark, get_s)
    ->MinWarmUpTime(1.0)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Ranges({{2048, 8192}, {4, 16}});

BENCHMARK_DEFINE_F(DR_BCG_Benchmark, get_sigma)(benchmark::State &state)
{
    const int m = state.range(0);
    const int n = state.range(1);
    auto [d_A, d_X, d_B] = initialize_inputs(m, n);

    DeviceBuffer d = filled_device_buffer(cusolverH, cusolverParams, cublasH, m, n, d_A, d_X, d_B);
    CUDA_CHECK(cudaDeviceSynchronize());

    dr_bcg::get_xi(cusolverH, cusolverParams, cublasH, m, n, d, d_A);
    dr_bcg::get_next_X(cublasH, m, n, d.s, d.xi, d.temp, d.sigma, d_X);
    dr_bcg::get_w_zeta(cusolverH, cusolverParams, cublasH, m, n, d, d_A);
    dr_bcg::get_s(cublasH, m, n, d);

    // Keep copies of zeta and sigma for consistent benchmark state
    std::vector<float> h_zeta(n * n);
    std::vector<float> h_sigma(n * n);
    CUDA_CHECK(cudaMemcpy(h_zeta.data(), d.zeta, sizeof(float) * h_zeta.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sigma.data(), d.sigma, sizeof(float) * h_sigma.size(), cudaMemcpyDeviceToHost));
    
    for (auto _ : state)
    {
        CUDA_CHECK(cudaMemcpy(d.zeta, h_zeta.data(), sizeof(float) * h_zeta.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.sigma, h_sigma.data(), sizeof(float) * h_sigma.size(), cudaMemcpyHostToDevice));
        TIME_CUDA(dr_bcg::get_sigma(cublasH, n, d));
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_B));
}
BENCHMARK_REGISTER_F(DR_BCG_Benchmark, get_sigma)
    ->MinWarmUpTime(1.0)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Ranges({{2048, 8192}, {4, 16}});

BENCHMARK_DEFINE_F(QR_Benchmark, qr_factorization)(benchmark::State &state)
{
    const int m = state.range(0);
    const int n = state.range(1);

    std::vector<float> h_A(m * n);
    fill_random(h_A.data(), m, n);

    float *d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(float) * h_A.size()));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeof(float) * h_A.size(), cudaMemcpyHostToDevice));

    float *d_Q = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Q, sizeof(float) * m * n));

    float *d_R = nullptr;
    CUDA_CHECK(cudaMalloc(&d_R, sizeof(float) * n * n));

    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnParams_t cusolverParams = NULL;

    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUSOLVER_CHECK(cusolverDnCreateParams(&cusolverParams));

    for (auto _ : state)
    {
        TIME_CUDA(dr_bcg::qr_factorization(cusolverH, cusolverParams, d_Q, d_R, m, n, d_A));
    }
}
BENCHMARK_REGISTER_F(QR_Benchmark, qr_factorization)
    ->MinWarmUpTime(1.0)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Ranges({{2048, 8192}, {4, 16}});

BENCHMARK_DEFINE_F(QR_Benchmark, thin_qr)(benchmark::State &state)
{
    const int m = state.range(0);
    const int n = state.range(1);

    std::vector<float> h_A(m * n);
    fill_random(h_A.data(), m, n);

    float *d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(float) * h_A.size()));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeof(float) * h_A.size(), cudaMemcpyHostToDevice));

    float *d_Q = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Q, sizeof(float) * m * n));

    float *d_R = nullptr;
    CUDA_CHECK(cudaMalloc(&d_R, sizeof(float) * n * n));

    cublasHandle_t cublasH = NULL;
    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnParams_t cusolverParams = NULL;

    CUBLAS_CHECK(cublasCreate_v2(&cublasH));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUSOLVER_CHECK(cusolverDnCreateParams(&cusolverParams));

    for (auto _ : state)
    {
        TIME_CUDA(dr_bcg::thin_qr(cusolverH, cusolverParams, cublasH, d_Q, d_R, m, n, d_A));
    }
}
BENCHMARK_REGISTER_F(QR_Benchmark, thin_qr)
    ->MinWarmUpTime(1.0)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Ranges({{2048, 8192}, {4, 16}});