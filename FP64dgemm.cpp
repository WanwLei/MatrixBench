#include <CL/sycl.hpp>
namespace sycl = cl::sycl;
using namespace sycl;

void mm_kernel(queue &q,
               std::vector<double> &matrix_a,
               std::vector<double> &matrix_b,
               std::vector<double> &matrix_c,
               size_t N, size_t M) {

    std::cout <<"Configuration: MATRIX SIZE = " << N << "x" << N
              << " | TILE SIZE = " << M << "x" << M << "\n";

    // --- Create buffers for full N*N matrices ---
    buffer a(matrix_a.data(), range<1>(N*N));
    buffer b(matrix_b.data(), range<1>(N*N));
    buffer c(matrix_c.data(), range<1>(N*N));

    // --- padded global size: must be multiple of M ---
    size_t G = ((N + M - 1) / M) * M;

    // --- Submit command group ---
    auto e = q.submit([&](handler &h){
        // --- Accessors ---
        accessor A(a, h, read_only);
        accessor B(b, h, read_only);
        accessor C(c, h, write_only, no_init);

        // --- Local (shared memory) tiles ---
        accessor<double, 2, access::mode::read_write, access::target::local> 
            A_tile(range<2>(M, M), h);
        accessor<double, 2, access::mode::read_write, access::target::local> 
            B_tile(range<2>(M, M), h);

        // --- Define nd_range ---
        nd_range<2> ndRange({G, G}, {M, M});

        h.parallel_for(ndRange, [=](nd_item<2> item){
            const int i = item.get_global_id(0);
            const int j = item.get_global_id(1);
            const int x = item.get_local_id(0);
            const int y = item.get_local_id(1);

            double temp = 0.0;

            // loop over tiles
            for (int t = 0; t < N; t += M) {
                // load into local tiles with boundary check
                int a_col = t + y;
                int b_row = t + x;

                A_tile[x][y] = (i < N && a_col < N) ? A[i * N + a_col] : 0.0;
                B_tile[x][y] = (b_row < N && j < N) ? B[b_row * N + j] : 0.0;

                item.barrier(access::fence_space::local_space);

                for (int k = 0; k < M; k++) {
                    temp += A_tile[x][k] * B_tile[k][y];
                }

                item.barrier(access::fence_space::local_space);
            }

            if (i < N && j < N) {
                C[i*N + j] = temp;
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
    size_t M = 32;    // tile 尺寸（可改成 8, 16, 32 测试性能）

    // 分配 N*N 大小的矩阵
    std::vector<double> matrix_a(N*N, 1.0);
    std::vector<double> matrix_b(N*N, 1.0);
    std::vector<double> matrix_c(N*N, 0.0);

    sycl::property_list propList{sycl::property::queue::enable_profiling()};
    queue Q(gpu_selector{}, propList);

    mm_kernel(Q, matrix_a, matrix_b, matrix_c, N, M);

    // 验证几个值（全 1 相乘，结果应该都是 N）
    std::cout << "C[0], C[N*N/2], C[N*N-1] = "
              << matrix_c[0] << ", "
              << matrix_c[(N*N)/2] << ", "
              << matrix_c[N*N-1] << "\n";

    return 0;
}
