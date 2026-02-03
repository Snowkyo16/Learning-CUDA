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


// FlashAttention Kernel: 使用 Online Softmax 算法
// 每个线程负责计算一个输出元素 O[b, t, h, d]
template <typename T>
__global__ void flashAttentionKernel(
    const T* Q,  const T* K, const T* V, T* O,
    int batch_size, int tgt_seq_len, int src_seq_len, 
    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  
  // 计算总输出元素数，每个线程处理一个
  int total = batch_size * tgt_seq_len * query_heads * head_dim;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= total) return;

  // 从线性索引反推 4D 坐标 [b, t, h, d]
  int d = idx % head_dim;
  int h = (idx / head_dim) % query_heads;
  int t = (idx / head_dim / query_heads) % tgt_seq_len;
  int b = idx / head_dim / query_heads / tgt_seq_len;

  // GQA 映射
  // 多个 query head 共享同一个 kv head
  int group_size = query_heads / kv_heads;
  int kv_h = h / group_size;

  // Online Softmax 状态变量
  float m_i = -INFINITY;  // 当前最大 score
  float l_i = 0.0f;        // softmax 分母累积
  float o_i = 0.0f;        // 加权输出累积

  // 遍历所有 key/value 位置
  for (int s = 0; s <  src_seq_len; s++) {
    // Causal mask: 只能看见当前位置及之前的 token
    if (is_causal &&  s > t) continue;

    // 计算 score = q · k / sqrt(head_dim)
    float score = 0.0f;
    for (int i = 0; i < head_dim; i++) {
      int q_idx = idx4d(b, t, h, i, tgt_seq_len, query_heads, head_dim);
      int k_idx = idx4d(b, s, kv_h, i, src_seq_len, kv_heads, head_dim);
      score += toFloat(Q[q_idx]) * toFloat(K[k_idx]);
    }
    score /= sqrtf((float)head_dim);
  
    // 获取 V[b, s, kv_h, d]的值
    int v_idx = idx4d(b, s, kv_h, d, src_seq_len, kv_heads, head_dim);
    float v = toFloat(V[v_idx]);

    // Online Softmax 更新
    float m_prev = m_i;
    m_i = fmax(m_i, score);  // 更新最大值
    float correction = expf(m_prev - m_i);  // 之前累积的修正因子
    float  weight = expf(score - m_i);  // 当前位置的权重
    l_i = l_i * correction + weight;    // 更新分母
    o_i = o_i * correction + weight * v;  // 更新加权和
  }

  // 最终输出 = 加权和 / 分母
  if (l_i > 0.0f) {
    O[idx] = fromFloat<T>(o_i / l_i);
  } else {
    O[idx] = fromFloat<T>(0.0f);
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
  int threads = 256;
  int blocks = (o_size + threads - 1) / threads;
  flashAttentionKernel<T><<<blocks, threads>>>(
    d_q, d_k, d_v, d_o,
    batch_size, target_seq_len, src_seq_len,
    query_heads, kv_heads, head_dim, is_causal);

  // 4. GPU -> CPU 结果拷贝
  h_o.resize(o_size);
  cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost);

  // 5. 适当 GPU 内存
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
