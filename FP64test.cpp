#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include <chrono>

namespace sycl = cl::sycl;
using namespace sycl;
int main() {
    const size_t N = 32768;  // 矩阵维度

    std::vector<double> A(N * N, 1.0);
    std::vector<double> B(N * N, 2.0);
    std::vector<double> C(N * N, 0.0);

    sycl::queue q(sycl::gpu_selector{});
    auto dev = q.get_device();

    std::cout << "Using device: " 
              << dev.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Vendor: " 
              << dev.get_info<sycl::info::device::vendor>() << std::endl;
    std::cout << "Driver: " 
              << dev.get_info<sycl::info::device::driver_version>() << std::endl;

    size_t max_compute_units = dev.get_info<sycl::info::device::max_compute_units>();
    size_t max_work_group = dev.get_info<sycl::info::device::max_work_group_size>();
    size_t max_freq = dev.get_info<sycl::info::device::max_clock_frequency>(); // MHz
    size_t global_mem = dev.get_info<sycl::info::device::global_mem_size>();

    std::cout << "Compute Units: " << max_compute_units << std::endl;
    std::cout << "Max Work-group size: " << max_work_group << std::endl;
    std::cout << "Max Clock Frequency: " << max_freq << " MHz" << std::endl;
    std::cout << "Global Memory: " << (global_mem >> 20) << " MB" << std::endl;

    // 理论峰值估算（每CU=64 ALUs，每ALU每周期1 FMA=2 FLOPs）
    double freq_Hz = max_freq * 1e6;
    double dp_flops_per_cycle_per_cu = 64 * 2 * 0.5; 
    // 双精度是单精度的一半
    double peak_dp = max_compute_units * dp_flops_per_cycle_per_cu * freq_Hz;
    double peak_dp_tflops = peak_dp / 1e12;

    std::cout << "Estimated peak FP64 performance: "
              << peak_dp_tflops << " TFLOP/s" << std::endl;

    // =================== 计时矩阵乘 ===================
    {
        buffer<double, 2> bufA(A.data(), range<2>(N, N));
        buffer<double, 2> bufB(B.data(), range<2>(N, N));
        buffer<double, 2> bufC(C.data(), range<2>(N, N));

        // ------------------ 数据拷贝时间 ------------------
        auto start_copy = std::chrono::high_resolution_clock::now();
        {
            host_accessor accA(bufA, write_only);
            host_accessor accB(bufB, write_only);
        } // 确保数据写入 device buffer
        auto end_copy_in = std::chrono::high_resolution_clock::now();

        // ------------------ 核函数执行时间 ------------------
        auto start_kernel = std::chrono::high_resolution_clock::now();
        q.submit([&](handler &h) {
            accessor a(bufA, h, read_only);
            accessor b(bufB, h, read_only);
            accessor c(bufC, h, write_only, no_init);

            h.parallel_for(range<2>(N, N), [=](id<2> idx) {
                double sum = 0.0;
                for (size_t k = 0; k < N; k++) {
                    sum += a[idx[0]][k] * b[k][idx[1]];
                }
                c[idx] = sum;
            });
        }).wait();
        auto end_kernel = std::chrono::high_resolution_clock::now();

        // ------------------ 回传数据时间 ------------------
        auto start_copy_out = std::chrono::high_resolution_clock::now();
        {
            host_accessor accC(bufC, read_only);
        } // 数据拷回 host
        auto end_copy_out = std::chrono::high_resolution_clock::now();

        // 统计 FLOPs
        double flops = 2.0 * N * N * N;  // 每个元素 N 次乘加 = 2N³ FLOPs
        double kernel_time = std::chrono::duration<double>(end_kernel - start_kernel).count();
        double copy_in_time = std::chrono::duration<double>(end_copy_in - start_copy).count();
        double copy_out_time = std::chrono::duration<double>(end_copy_out - start_copy_out).count();

        std::cout << "Matrix size: " << N << "x" << N << "\n";
        std::cout << "Kernel execution time: " << kernel_time << " s\n";
        std::cout << "Copy to device time:   " << copy_in_time << " s\n";
        std::cout << "Copy to host time:     " << copy_out_time << " s\n";
        std::cout << "Total time:            " << (kernel_time + copy_in_time + copy_out_time) << " s\n";
        std::cout << "Achieved performance:  " << (flops / kernel_time / 1e9) << " GFLOP/s (kernel only)\n";
        std::cout << "Achieved performance:  " << (flops / (kernel_time + copy_in_time + copy_out_time) / 1e9) << " GFLOP/s (with copy)\n";

        std::cout << "C[0][0] = " << C[0] << "\n";
    }

    return 0;
}