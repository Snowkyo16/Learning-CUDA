#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
// GPU Kernel: 提取对角线元素
template <typename T>
__global__ void traceKernel(const T* d_input, T* d_partial, int cols, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    d_partial[i] = d_input[i * cols + i];
  }
}

// 主函数
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // TODO: Implement the trace function
  int n = (rows < cols) ? rows : cols;  // min(rows, cols) 对角线个数
  if (n == 0) {
    return T(0);
  }

  // 1. GPU内存分配
  T* d_input;
  T* d_partial;
  cudaMalloc(&d_input, rows * cols * sizeof(T));
  cudaMalloc(&d_partial, n * sizeof(T));

  // 2. CPU -> GPU
  cudaMemcpy(d_input, h_input.data(), rows * cols * sizeof(T), cudaMemcpyHostToDevice);

  // 3. 启动 kernel
  int threads = 256;
  int blocks = (n + threads - 1) / threads;  // 向上取整
  traceKernel<<<blocks, threads>>>(d_input, d_partial, cols, n);

  // 4. GPU -> CPU
  std::vector<T> h_partial(n);
  cudaMemcpy(h_partial.data(), d_partial, n * sizeof(T), cudaMemcpyDeviceToHost);

  // 5. CPU 求和
  T sum = 0;
  for (int i = 0; i < n; i++) {
    sum += h_partial[i];
  }

  // 6. 释放内存
  cudaFree(d_input);
  cudaFree(d_partial);

  return sum;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */

// 计算 4D 索引，行主序存储
 __device__ __forceinline__  int idx4d(int i0, int i1, int i2, int i3,
                                       int D1, int  D2, int D3) {
  return i0 * (D1 * D2 *  D3) + i1 * (D2 * D3) + i2 * D3 + i3;
}

// 类型转换：统一转为 float 计算，避免 half 精度问题
template <typename T>
__device__ __forceinline__ float toFloat(T x) {
  return (float) x;
}

template <>
__device__ __forceinline__ float toFloat<half>(half x) {
  return __half2float(x);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float x) {
  return (T)x;
}

template <>
__device__ __forceinline__ half fromFloat<half>(float x) {
  return __float2half(x);
}

#define MAX_HEAD_DIM 256

// FlashAttention Kernel: 块级并行 + 共享内存缓存 K
// 每个 block 处理一个 [b, t, h]，每个线程负责一个 d 维度
template <typename T>
__global__ void flashAttentionKernel(
    const T* Q,  const T* K, const T* V, T* O,
    int batch_size, int tgt_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim, bool is_causal) {

  // 从 block 索引计算 [b, t, h]
  int block_idx = blockIdx.x;
  int h = block_idx % query_heads;
  int t = (block_idx / query_heads) % tgt_seq_len;
  int b = block_idx / query_heads / tgt_seq_len;

  // 每个线程负责一个 d 维度
  int tid = threadIdx.x;

  // GQA 映射
  int group_size = query_heads / kv_heads;
  int kv_h = h / group_size;

  // 共享内存缓存 K 向量
  __shared__ float K_shared[MAX_HEAD_DIM];

  // 预计算基地址
  int q_base = idx4d(b, t, h, 0, tgt_seq_len, query_heads, head_dim);
  int kv_stride_s = kv_heads * head_dim;
  int kv_base = b * src_seq_len * kv_stride_s + kv_h * head_dim;

  // Causal mask 边界
  int s_end = is_causal ? (t + 1 < src_seq_len ? t + 1 : src_seq_len) : src_seq_len;

  // Online Softmax 状态变量
  float m_i = -INFINITY;
  float l_i = 0.0f;
  float o_i = 0.0f;

  // 遍历所有 key/value 位置
  for (int s = 0; s < s_end; s++) {
    int kv_offset = kv_base + s * kv_stride_s;

    // 1. 协作加载 K 到共享内存
    if (tid < head_dim) {
      K_shared[tid] = toFloat(K[kv_offset + tid]);
    }
    __syncthreads();

    // 2. 每个线程独立顺序计算完整 score（保持累加顺序一致）
    float score = 0.0f;
    for (int i = 0; i < head_dim; i++) {
      score += toFloat(Q[q_base + i]) * K_shared[i];
    }
    score /= sqrtf((float)head_dim);

    // 3. Online Softmax 更新
    float m_prev = m_i;
    m_i = fmaxf(m_i, score);
    float correction = expf(m_prev - m_i);
    float weight = expf(score - m_i);
    l_i = l_i * correction + weight;

    // 4. 累加 V 的贡献
    if (tid < head_dim) {
      float v = toFloat(V[kv_offset + tid]);
      o_i = o_i * correction + weight * v;
    }
    __syncthreads();
  }

  // 最终输出 = 加权和 / 分母
  if (tid < head_dim) {
    int o_idx = idx4d(b, t, h, tid, tgt_seq_len, query_heads, head_dim);
    O[o_idx] = fromFloat<T>(l_i > 0.0f ? (o_i / l_i) : 0.0f);
  }
}


template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  // TODO: Implement the flash attention function
  // 计算各 tensor 大小
  int q_size = batch_size * target_seq_len * query_heads * head_dim;
  int kv_size = batch_size * src_seq_len * kv_heads * head_dim;
  int o_size = q_size;

  // 1. GPU 内存分配
  T *d_q, *d_k, *d_v, *d_o;
  cudaMalloc(&d_q, q_size * sizeof(T));
  cudaMalloc(&d_k, kv_size * sizeof(T));
  cudaMalloc(&d_v, kv_size * sizeof(T));
  cudaMalloc(&d_o, o_size * sizeof(T));

  //  2. CPU -> GPU
  cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_k.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice);

  // 3. 启动kernel
  int num_blocks = batch_size * target_seq_len * query_heads;
  int threads_per_block = std::max(head_dim, 32);
  threads_per_block = std::min(threads_per_block, 256);
  flashAttentionKernel<T><<<num_blocks, threads_per_block>>>(
    d_q, d_k, d_v, d_o,
    batch_size, target_seq_len, src_seq_len,
    query_heads, kv_heads, head_dim, is_causal);

  // 4. GPU -> CPU 结果拷贝
  h_o.resize(o_size);
  cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost);

  // 5. 释放 GPU 内存
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
