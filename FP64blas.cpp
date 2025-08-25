#include <CL/sycl.hpp>
#include <rocblas.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <limits>

namespace sycl = cl::sycl;

static void check_rocblas(rocblas_status st, const char* msg) {
    if (st != rocblas_status_success) {
        std::cerr << "[rocBLAS] " << msg << " failed, status = " << st << std::endl;
        std::exit(1);
    }
}

int main() {
    const size_t N = 4096;                // 
    const double alpha = 1.0, beta = 0.0; // C = alpha*A*B + beta*C

    // -------- Host matrices (row-major) --------
    std::vector<double> A(N * N, 1.0); // row-major
    std::vector<double> B(N * N, 1.0); // row-major
    std::vector<double> C(N * N, 0.0); // row-major

    // -------- SYCL queue --------
    sycl::queue q(sycl::gpu_selector{});
    auto dev = q.get_device();
    std::cout << "Using device: "
              << dev.get_info<sycl::info::device::name>() << std::endl;

    // -------- rocBLAS handle --------
    rocblas_handle handle = nullptr;
    check_rocblas(rocblas_create_handle(&handle), "rocblas_create_handle");

    // 确认/获取 rocBLAS 使用的 HIP stream，后面用它做同步
    hipStream_t rb_stream = nullptr;
    check_rocblas(rocblas_get_stream(handle, &rb_stream), "rocblas_get_stream");
    if (!rb_stream) {
        // 如果为空，rocBLAS会使用默认流；这里不强制绑定到 SYCL 的底层流，直接用 HIP 同步即可
        rb_stream = nullptr;
    }

    // -------- Device memory (USM device) --------
    auto bytes = sizeof(double) * N * N;
    double* dA = sycl::malloc_device<double>(N * N, q);
    double* dB = sycl::malloc_device<double>(N * N, q);
    double* dC = sycl::malloc_device<double>(N * N, q);
    if (!dA || !dB || !dC) {
        std::cerr << "Device allocation failed." << std::endl;
        return 1;
    }

    // -------- Copy H2D --------
    auto t0 = std::chrono::high_resolution_clock::now();
    q.memcpy(dA, A.data(), bytes).wait();
    q.memcpy(dB, B.data(), bytes).wait();
    q.memset(dC, 0, bytes).wait();
    auto t1 = std::chrono::high_resolution_clock::now();

    // -------- GEMM (row-major ) --------
    // rocBLAS 计算 C_col = B_col * A_col = (A_row * B_row)^T
    // 这样把 A/B 位置对调（均不转置），拷回后按 row-major 访问就是正确的 C
    if (N > static_cast<size_t>(std::numeric_limits<int>::max())) {
        std::cerr << "N is too large for rocBLAS int parameters." << std::endl;
        return 1;
    }
    const int n = static_cast<int>(N);

    auto t2 = std::chrono::high_resolution_clock::now();
    rocblas_status st = rocblas_dgemm(
        handle,
        rocblas_operation_none,   // op(B)  B放在第一个操作数位置
        rocblas_operation_none,   // op(A)
        n,                        // m
        n,                        // n
        n,                        // k
        &alpha,
        dB, n,                    // A指针位置传 dB，lda = N
        dA, n,                    // B指针位置传 dA，ldb = N
        &beta,
        dC, n                     // C (col-major)
    );
    check_rocblas(st, "rocblas_dgemm");

    // 同步：确保 GEMM 真正完成（rocBLAS 异步）
    if (rb_stream) hipStreamSynchronize(rb_stream);
    else hipDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();

    // -------- Copy D2H --------
    q.memcpy(C.data(), dC, bytes).wait();
    auto t4 = std::chrono::high_resolution_clock::now();

    // -------- Timing & FLOPs --------
    double copy_in_s  = std::chrono::duration<double>(t1 - t0).count();
    double gemm_s     = std::chrono::duration<double>(t3 - t2).count();
    double copy_out_s = std::chrono::duration<double>(t4 - t3).count();
    double flops      = 2.0 * N * N * N;

    std::cout << "Matrix size: " << N << " x " << N << "\n";
    std::cout << "Copy to device: " << copy_in_s  << " s\n";
    std::cout << "GEMM time:      " << gemm_s     << " s\n";
    std::cout << "Copy to host:   " << copy_out_s << " s\n";
    std::cout << "Total:          " << (copy_in_s + gemm_s + copy_out_s) << " s\n";
    std::cout << "Perf (kernel):  " << (flops / gemm_s / 1e12) << " TFLOP/s\n";
    std::cout << "Perf (w/ copy): " << (flops / (copy_in_s + gemm_s + copy_out_s) / 1e12) << " TFLOP/s\n";

    // -------- Quick check --------
    // A=1, B=1 => C 全是 N
    std::cout << "C[0], C[N*N/2], C[N*N-1] = "
              << C[0] << ", "
              << C[(N*N)/2] << ", "
              << C[N*N-1] << "\n";

    // -------- Cleanup --------
    rocblas_destroy_handle(handle);
    sycl::free(dA, q);
    sycl::free(dB, q);
    sycl::free(dC, q);

    return 0;
}
