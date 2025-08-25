#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include <chrono>

namespace sycl = cl::sycl;
using namespace sycl;

void mm_kernel(queue &q,
               std::vector<float> &matrix_a,
               std::vector<float> &matrix_b,
               std::vector<float> &matrix_c,
               size_t N, size_t M) {

    std::cout <<"Configuration: MATRIX SIZE = " << N << "x" << N
              << " | TILE SIZE = " << M << "x" << M << "\n";

    // --- 在 host 端先把 B 转置成 BT ---
    std::vector<float> matrix_bT(N * N);
    for (size_t r = 0; r < N; ++r) {
        const size_t rN = r * N;
        for (size_t c = 0; c < N; ++c) {
            matrix_bT[c * N + r] = matrix_b[rN + c];
        }
    }

    // --- Create buffers for full N*N matrices ---
    buffer<float, 1> a(matrix_a.data(), range<1>(N*N));
    buffer<float, 1> bt(matrix_bT.data(), range<1>(N*N)); 
    buffer<float, 1> c(matrix_c.data(), range<1>(N*N));

    // --- padded global size: must be multiple of M ---
    size_t G = ((N + M - 1) / M) * M;

    // --- Submit command group ---
    auto e = q.submit([&](handler &h){
        // --- Accessors ---
        accessor A(a, h, read_only);
        accessor BT(bt, h, read_only);
        accessor C(c, h, write_only, no_init);

        // --- Local (shared memory) tiles ---
        accessor<float, 2, access::mode::read_write, access::target::local> 
            A_tile(range<2>(M, M), h);
        accessor<float, 2, access::mode::read_write, access::target::local> 
            B_tile(range<2>(M, M), h);

        // --- Define nd_range ---
        nd_range<2> ndRange({G, G}, {M, M});

        h.parallel_for(ndRange, [=](nd_item<2> item){
            const int i = item.get_global_id(0); // row of C
            const int j = item.get_global_id(1); // col of C
            const int x = item.get_local_id(0);  // row within tile
            const int y = item.get_local_id(1);  // col within tile

            float temp = 0.0f;

            // loop over tiles
            for (int t = 0; t < (int)N; t += (int)M) {
                // ---- 加载到 local tiles（边界检查）----
                int a_col = t + y;
                A_tile[x][y] = (i < (int)N && a_col < (int)N) ? A[i * (int)N + a_col] : 0.0f;

                int bt_col = t + x;
                B_tile[x][y] = (j < (int)N && bt_col < (int)N) ? BT[j * (int)N + bt_col] : 0.0f;

                item.barrier(access::fence_space::local_space);

                // ---- 计算 tile 内乘加 ----
                for (int k = 0; k < (int)M; k++) {
                    temp += A_tile[x][k] * B_tile[k][y];
                }

                item.barrier(access::fence_space::local_space);
            }

            if (i < (int)N && j < (int)N) {
                C[i*(int)N + j] = temp;
            }
        });
    });

    // --- Ensure completion ---
    host_accessor hc(c, read_only);

    // --- Profiling ---
    auto kernel_duration =
        (e.get_profiling_info<info::event_profiling::command_end>() -
         e.get_profiling_info<info::event_profiling::command_start>());
    std::cout << "Kernel Execution Time: "
              << kernel_duration / 1e9 << " seconds\n";
}

int main() {
    size_t N = 32768;  // 矩阵维度
    size_t M = 32;    // tile 尺寸（可改 8,16,32 测性能）

    std::vector<float> matrix_a(N*N, 1.0f);
    std::vector<float> matrix_b(N*N, 1.0f);
    std::vector<float> matrix_c(N*N, 0.0f);

    sycl::property_list propList{sycl::property::queue::enable_profiling()};
    queue Q(gpu_selector{}, propList);

    mm_kernel(Q, matrix_a, matrix_b, matrix_c, N, M);

    // 全 1 相乘，结果应为 N
    std::cout << "C[0], C[N*N/2], C[N*N-1] = "
              << matrix_c[0] << ", "
              << matrix_c[(N*N)/2] << ", "
              << matrix_c[N*N-1] << "\n";

    return 0;
}
