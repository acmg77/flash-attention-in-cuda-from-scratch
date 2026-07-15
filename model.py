"""
Flash Attention in CUDA from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - vector_add
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // TODO: implement elementwise c[i] = a[i] + b[i]
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid<n){
        c[gid] = a[gid] + b[gid];
    }
}

# Step 2 - scale_array
__global__ void scale_array(float* a, float scalar, int n) {
    // TODO: multiply each element of a by scalar in place
    int gid = blockIdx.x*blockDim.x + threadIdx.x;

    if(gid<n){
        a[gid]*=scalar;
    }
}

# Step 3 - elementwise_exp
__global__ void elementwise_exp(float* a, int n) {
    // TODO: replace each a[i] with expf(a[i])
    int gid = blockIdx.x*blockDim.x + threadIdx.x;

    if(gid<n){
        a[gid] = expf(a[gid]);
    }
}

# Step 4 - row_max
__global__ void row_max(const float* matrix, float* out, int rows, int cols) {
    // TODO: compute the max of each row and write it to out[r].
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i = gid;i<rows;i+=stride){
        float cur_max = matrix[i*cols];
        for(int c = 1;c<cols;++c){
            if(matrix[i*cols+c]>cur_max){
                cur_max = matrix[i*cols+c];
            }
        }
        out[i] = cur_max;
    }
}

# Step 5 - row_sum
__global__ void row_sum(const float* matrix, float* out, int rows, int cols) {
    // TODO: write out[r] = sum of matrix row r
    int gid = blockIdx.x*blockDim.x + threadIdx.x;

    int stride = gridDim.x * blockDim.x;

    for(int i = gid;i<rows;i+=stride){
        float sum = matrix[i*cols];
        for(int c = 1;c<cols;++c){
            sum+=matrix[i*cols+c];
        }
        out[i] = sum;
    }
}

# Step 6 - dot_product
__device__ float dot_product(const float* a, const float* b, int n) {
    // TODO: return the dot product of a and b
    float sum = 0.0f;
    
    // 每个线程顺序计算传入数组的点积
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    
    return sum;
}

# Step 7 - matmul
__global__ void matmul(const float* a, const float* b, float* c, int m, int k, int n) {
    // TODO: compute C = A * B for row-major matrices
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride: total number of threads in the grid
    int stride = gridDim.x * blockDim.x;

    int num = m*n;
    for(int i = gid;i<num;i+=stride){
        int col = i%n;
        int row = i/n;
        float sum = 0.0f;
        for(int c = 0;c<k;++c){
            sum+=a[row*k+c]*b[c*n+col];
        }
        c[i] = sum;

    }
}

# Step 8 - transpose
__global__ void transpose(const float* in, float* out, int rows, int cols) {
    // TODO: write out[c*rows + r] = in[r*cols + c]
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int num = rows*cols;
    

    for(int i = gid;i<num;i+=stride){
        int new_cols = i%cols;
        int new_rows = i/cols;
        // for(int c = 0;c<cols;++c){
        //     out[new_rows+c*new_cols] = in[i*cols+c];
        // }
        out[new_rows+rows*new_cols] = in[i];
    }

}

# Step 9 - qk_scores
__global__ void qk_scores(const float* q, const float* k, float* scores, int seq_len, int head_dim) {
    // TODO: compute scores[i, j] = dot(q_row_i, k_row_j) / sqrt(head_dim)
    // int scale = 1.0/sqrt(head_dim);
    int j = blockIdx.x * blockDim.x + threadIdx.x; // 输出矩阵的列索引
    int i = blockIdx.y * blockDim.y + threadIdx.y; // 输出矩阵的行索引

    // 2. 边界检查，防止越界访问 (输出矩阵的形状是 seq_len x seq_len)
    if (i < seq_len && j < seq_len) {
        
        // 3. 计算缩放因子 (注意这里必须是 float 类型，并且推荐用 sqrtf 计算单精度)
        float scale = 1.0f / sqrtf((float)head_dim);

        // 4. 定位 Q 和 K 中对应行的起始指针
        // 矩阵是行优先 (row-major) 存储的，形状为 (seq_len, head_dim)
        const float* q_row_i = q + i * head_dim;
        const float* k_row_j = k + j * head_dim;

        // 5. 调用外部的 dot_product 计算点积，并乘上缩放因子，写入输出矩阵
        // 输出矩阵 scores 也是行优先存储，形状为 (seq_len, seq_len)
        scores[i * seq_len + j] = dot_product(q_row_i, k_row_j, head_dim) * scale;
    }

}

# Step 10 - softmax_rows
__global__ void softmax_rows(float* matrix, int rows, int cols) {
    // TODO: implement numerically stable row-wise softmax in place
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    // int stride = ;
    if (gid < rows) {
        // 定位当前行的起始指针，让代码更清爽，减少冗余的乘法计算
        float* row_ptr = matrix + gid * cols;
        
        // 1. 第一趟：寻找当前行的最大值 max_num
        float max_num = row_ptr[0];
        for (int c = 1; c < cols; ++c) {
            if (row_ptr[c] > max_num) {
                max_num = row_ptr[c];
            }
        }
        
        // 2. 第二趟：计算数值稳定的指数和 sum
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            sum += expf(row_ptr[c] - max_num);
        }
        
        // 3. 第三趟：计算最终的 Softmax 并原地 (in-place) 写回
        for (int c = 0; c < cols; ++c) {
            row_ptr[c] = expf(row_ptr[c] - max_num) / sum;
        }
    }
}

# Step 11 - pv_matmul
__global__ void pv_matmul(const float* p, const float* v, float* out, int seq_len, int head_dim) {
    // TODO: compute out[i, d] = sum_j p[i, j] * v[j, d]
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    int i = blockIdx.y*blockDim.y+threadIdx.y;

    if(i<seq_len&&j<head_dim){
        //out_dim = [seq_len,head_dim]
        float sum = 0.0f;
        for(int c = 0;c<seq_len;++c){
            sum += p[i*seq_len+c]*v[c*head_dim+j];
        }
        out[i*head_dim+j] = sum;
    }
}

# Step 12 - naive_attention
void naive_attention(const float* d_q, const float* d_k, const float* d_v, float* d_out, int seq_len, int head_dim) {
    // TODO: allocate scratch, launch qk_scores -> softmax_rows -> pv_matmul, free scratch
    float* d_p = nullptr;
    
    // 1. Allocate scratch space for the attention scores / probabilities.
    // Dimensions: [seq_len, seq_len]
    size_t p_size = (size_t)seq_len * seq_len * sizeof(float);
    cudaError_t err = cudaMalloc(&d_p, p_size);
    if (err != cudaSuccess) {
        // 在生产环境中应当有更好的错误处理机制
        return;
    }

    // 2. Launch qk_scores: computes d_p = (Q @ K^T) / sqrt(head_dim)
    // Q is [seq_len, head_dim], K is [seq_len, head_dim]
    // Output d_p is [seq_len, seq_len]
    dim3 block_qk(16, 16);
    dim3 grid_qk((seq_len + block_qk.x - 1) / block_qk.x, 
                 (seq_len + block_qk.y - 1) / block_qk.y);
    
    qk_scores<<<grid_qk, block_qk>>>(d_q, d_k, d_p, seq_len, head_dim);

    // 3. Launch softmax_rows: computes row-wise softmax on d_p in-place
    // Each row of d_p is independent. 
    // 假设这是一个简单的 1D kernel，每个线程（或每个 Block）处理一行数据
    int threads_per_block = 256;
    int blocks_per_grid = (seq_len + threads_per_block - 1) / threads_per_block;
    
    softmax_rows<<<blocks_per_grid, threads_per_block>>>(d_p, seq_len,seq_len);

    // 4. Launch pv_matmul: computes d_out = d_p @ V
    // d_p is [seq_len, seq_len], V is [seq_len, head_dim]
    // Output d_out is [seq_len, head_dim]
    dim3 block_pv(16, 16);
    dim3 grid_pv((head_dim + block_pv.x - 1) / block_pv.x, 
                 (seq_len + block_pv.y - 1) / block_pv.y);
                 
    pv_matmul<<<grid_pv, block_pv>>>(d_p, d_v, d_out, seq_len, head_dim);

    // 等待所有 Kernel 执行完毕（如果需要严格同步或计时可以加上）
    // cudaDeviceSynchronize();

    // 5. Free scratch memory
    cudaFree(d_p);
}

# Step 13 - online_max
__device__ float online_max(float old_max, float new_val) {
    // TODO: return the running max of old_max and new_val
    
    return max(old_max,new_val);
}

# Step 14 - correction_factor
__device__ float correction_factor(float old_max, float new_max) {
    // TODO: return the scalar used to rescale running statistics

    return expf(old_max-new_max);
}

# Step 15 - update_running_sum
__device__ float update_running_sum(float old_sum, float correction, float block_sum) {
    // TODO: combine the rescaled old sum with the new block sum

    return old_sum*correction+block_sum;
}

# Step 16 - rescale_output
__device__ void rescale_output(float* out_row, int head_dim, float correction) {
    // TODO: multiply each of the head_dim entries of out_row by correction in place
    for (int i = 0; i < head_dim; ++i) {
        out_row[i] *= correction;
    }
    
}

# Step 17 - load_tile
__device__ void load_tile(const float* src, float* shared_dst,
                          int src_row_start, int src_col_start,
                          int src_rows, int src_cols,
                          int tile_rows, int tile_cols,
                          int thread_id, int num_threads) {
    // TODO: cooperatively copy the tile into shared_dst, zero-filling out-of-bounds positions.
    int total_elements = tile_rows * tile_cols;
    for(int i = thread_id;i<total_elements;i+=num_threads){
        int matrix_row = i/tile_cols;
        int matrix_col = i%tile_cols;

        int global_row = src_row_start+matrix_row;
        int global_col = src_col_start+matrix_col;

        if(global_col<src_cols&&global_row<src_rows){
            shared_dst[matrix_col+matrix_row*tile_cols] = src[global_col+global_row*src_cols];
        }else{
            shared_dst[matrix_col+matrix_row*tile_cols] = 0.0f;
        }
    }
    
}

# Step 18 - tile_scores
__device__ void tile_scores(const float* q_tile, const float* k_tile, float* s_tile,
                            int tile_q, int tile_k, int head_dim, float scale,
                            int thread_id, int num_threads) {
    // TODO: cooperatively fill s_tile[i, j] = scale * dot(q_tile[i, :], k_tile[j, :])
    int total_num = tile_k*tile_q;
    for(int i = thread_id;i<total_num;i+=num_threads){
        int col = i % tile_k;
        int row = i / tile_k;
        float sum = 0.0f;
        // int g_col = q_tile+col;
        // int g_row = q_tile+row;

        for(int num=0;num<head_dim;++num){
            sum+=q_tile[num+row*head_dim]*k_tile[num+col*head_dim];
        }
        s_tile[i] = sum*scale;
    }
}

# Step 19 - tile_rowmax
__device__ void tile_rowmax(const float* s_tile, float* row_max_out, int tile_q, int tile_k, int thread_id, int num_threads) {
    // TODO: write row_max_out[r] = max over c of s_tile[r, c]
    for (int row = thread_id; row < tile_q; row += num_threads) {
        
        // 1. 初始化最大值为极小值（防止分数全为负数的情况）
        float max_val = -1e38f; 
        
        // 2. 当前线程独自遍历它所负责的这行的所有列
        for (int col = 0; col < tile_k; ++col) {
            float val = s_tile[row * tile_k + col];
            if (val > max_val) {
                max_val = val;
            }
        }
        
        // 3. 将这一行求出的最大值，安全地写入输出数组
        row_max_out[row] = max_val;
    }
}

# Step 20 - tile_exp
__device__ void tile_exp(float* s_tile, const float* row_max,
                         int tile_q, int tile_k,
                         int thread_id, int num_threads) {
    // TODO: for each (r, c) in the tile, set s_tile[r*tile_k+c] = expf(s_tile[r*tile_k+c] - row_max[r])
    const int num_elements = tile_q * tile_k;

    // 每个线程处理若干个展平后的元素
    for (int idx = thread_id; idx < num_elements; idx += num_threads) {
        const int row = idx / tile_k;

        s_tile[idx] = expf(s_tile[idx] - row_max[row]);
    }
}

# Step 21 - tile_rowsum
__device__ void tile_rowsum(const float* p_tile, float* row_sum_out,
                            int tile_q, int tile_k,
                            int thread_id, int num_threads) {
    for (int row = thread_id; row < tile_q; row += num_threads) {
        float sum = 0.0f;

        for (int col = 0; col < tile_k; ++col) {
            sum += p_tile[row * tile_k + col];
        }

        row_sum_out[row] = sum;
    }
}

# Step 22 - accumulate_pv
__device__ void accumulate_pv(const float* p_tile, const float* v_tile, float* out_acc, int tile_q, int tile_k, int head_dim, int thread_id, int num_threads) {
    // TODO: cooperatively add P_tile * V_tile into out_acc
    const int num_output_elements = tile_q * head_dim;

    for (int idx = thread_id;
         idx < num_output_elements;
         idx += num_threads) {

        // out_acc[row, col]
        const int row = idx / head_dim;
        const int col = idx % head_dim;

        float acc = 0.0f;

        // P 的第 row 行与 V 的第 col 列做点积
        for (int k = 0; k < tile_k; ++k) {
            acc += p_tile[row * tile_k + k]
                 * v_tile[k * head_dim + col];
        }

        // 注意这里是累加，而不是覆盖
        out_acc[row * head_dim + col] += acc;
    }
}

# Step 23 - flash_attention_kernel
__global__ void flash_attention_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int seq_len,
    int head_dim,
    int tile_q,
    int tile_k,
    float scale
) {
    extern __shared__ float shared_mem[];

    const int thread_id = threadIdx.x;
    const int num_threads = blockDim.x;

    // 一个 block 处理一个 Q tile
    const int q_start = blockIdx.x * tile_q;

    const int remaining_q = seq_len - q_start;
    const int valid_q =
        remaining_q < tile_q ? remaining_q : tile_q;

    if (valid_q <= 0) {
        return;
    }

    // ============================================================
    // Shared memory layout
    // ============================================================

    float* q_tile = shared_mem;
    // [tile_q, head_dim]

    float* k_tile =
        q_tile + tile_q * head_dim;
    // [tile_k, head_dim]

    float* v_tile =
        k_tile + tile_k * head_dim;
    // [tile_k, head_dim]

    float* p_tile =
        v_tile + tile_k * head_dim;
    // [tile_q, tile_k]
    // 先存 score，之后原地变为 exp(score - row_max)

    float* out_acc =
        p_tile + tile_q * tile_k;
    // [tile_q, head_dim]

    float* running_max =
        out_acc + tile_q * head_dim;
    // [tile_q]

    float* running_sum =
        running_max + tile_q;
    // [tile_q]

    float* block_stat =
        running_sum + tile_q;
    // [tile_q]
    // 先保存当前 block max，之后复用为 correction

    float* block_sum =
        block_stat + tile_q;
    // [tile_q]

    // ============================================================
    // 1. 加载当前 Q tile
    // ============================================================

    load_tile(
        q,
        q_tile,
        q_start,       // src_row_start
        0,             // src_col_start
        seq_len,       // src_rows
        head_dim,      // src_cols
        tile_q,        // tile_rows
        head_dim,      // tile_cols
        thread_id,
        num_threads
    );

    // 初始化输出累加器
    for (int index = thread_id;
         index < tile_q * head_dim;
         index += num_threads) {

        out_acc[index] = 0.0f;
    }

    // 初始化 online softmax 状态
    for (int row = thread_id;
         row < tile_q;
         row += num_threads) {

        if (row < valid_q) {
            running_max[row] = -INFINITY;
            running_sum[row] = 0.0f;
        } else {
            // 无效 Q 行只用于填充，避免后续产生 NaN
            running_max[row] = 0.0f;
            running_sum[row] = 1.0f;
        }

        block_stat[row] = 0.0f;
        block_sum[row] = 0.0f;
    }

    __syncthreads();

    // ============================================================
    // 2. 遍历所有 K/V tiles
    // ============================================================

    for (int kv_start = 0;
         kv_start < seq_len;
         kv_start += tile_k) {

        const int remaining_k = seq_len - kv_start;
        const int valid_k =
            remaining_k < tile_k ? remaining_k : tile_k;

        // --------------------------------------------------------
        // 2.1 加载 K tile
        // --------------------------------------------------------

        load_tile(
            k,
            k_tile,
            kv_start,      // src_row_start
            0,             // src_col_start
            seq_len,       // src_rows
            head_dim,      // src_cols
            tile_k,        // tile_rows
            head_dim,      // tile_cols
            thread_id,
            num_threads
        );

        // --------------------------------------------------------
        // 2.2 加载 V tile
        // --------------------------------------------------------

        load_tile(
            v,
            v_tile,
            kv_start,
            0,
            seq_len,
            head_dim,
            tile_k,
            head_dim,
            thread_id,
            num_threads
        );

        __syncthreads();

        // --------------------------------------------------------
        // 2.3 计算当前分块的 score
        //
        // P_tile = scale * Q_tile @ K_tile^T
        // --------------------------------------------------------

        tile_scores(
            q_tile,
            k_tile,
            p_tile,
            tile_q,
            tile_k,
            head_dim,
            scale,
            thread_id,
            num_threads
        );

        __syncthreads();

        // --------------------------------------------------------
        // 2.4 屏蔽最后一个不完整 tile 的无效位置
        //
        // load_tile 对越界 K 补零，但对应 score 会变成 0。
        // softmax 中这些位置必须是 -inf，不能是 0。
        // --------------------------------------------------------

        const int score_elements = tile_q * tile_k;

        for (int index = thread_id;
             index < score_elements;
             index += num_threads) {

            const int row = index / tile_k;
            const int col = index % tile_k;

            if (row >= valid_q || col >= valid_k) {
                p_tile[index] = -INFINITY;
            }
        }

        __syncthreads();

        // --------------------------------------------------------
        // 2.5 当前 score tile 的逐行最大值
        // --------------------------------------------------------

        tile_rowmax(
            p_tile,
            block_stat,
            tile_q,
            tile_k,
            thread_id,
            num_threads
        );

        __syncthreads();

        // --------------------------------------------------------
        // 2.6 更新 running max，并计算 correction
        //
        // new_max = max(old_max, block_max)
        // correction = exp(old_max - new_max)
        //
        // block_stat 原本保存 block_max，
        // 这里覆盖为 correction。
        // --------------------------------------------------------

        for (int row = thread_id;
             row < tile_q;
             row += num_threads) {

            if (row < valid_q) {
                const float old_max =
                    running_max[row];

                const float block_max =
                    block_stat[row];

                const float new_max =
                    online_max(old_max, block_max);

                const float correction =
                    correction_factor(old_max, new_max);

                running_max[row] = new_max;
                block_stat[row] = correction;
            } else {
                block_stat[row] = 0.0f;
            }
        }

        __syncthreads();

        // --------------------------------------------------------
        // 2.7 按行重新缩放历史输出
        //
        // out_acc[row] *= correction[row]
        //
        // rescale_output 的签名是：
        // rescale_output(float* out_row,
        //                int head_dim,
        //                float correction)
        // --------------------------------------------------------

        for (int row = thread_id;
             row < valid_q;
             row += num_threads) {

            rescale_output(
                out_acc + row * head_dim,
                head_dim,
                block_stat[row]
            );
        }

        __syncthreads();

        // --------------------------------------------------------
        // 2.8 计算当前 tile 的未归一化 softmax
        //
        // p_tile[row, col] =
        //     exp(score[row, col] - running_max[row])
        // --------------------------------------------------------

        tile_exp(
            p_tile,
            running_max,
            tile_q,
            tile_k,
            thread_id,
            num_threads
        );

        __syncthreads();

        // invalid K 列原本是 -inf，经过 exp 后自然为 0。
        // invalid Q 行也显式清零，避免依赖 inf 运算行为。
        for (int index = thread_id;
             index < score_elements;
             index += num_threads) {

            const int row = index / tile_k;

            if (row >= valid_q) {
                p_tile[index] = 0.0f;
            }
        }

        __syncthreads();

        // --------------------------------------------------------
        // 2.9 求当前 P tile 的逐行和
        // --------------------------------------------------------

        tile_rowsum(
            p_tile,
            block_sum,
            tile_q,
            tile_k,
            thread_id,
            num_threads
        );

        __syncthreads();

        // --------------------------------------------------------
        // 2.10 更新 online softmax 分母
        //
        // new_sum =
        //     old_sum * correction + block_sum
        //
        // 参数顺序必须是：
        // update_running_sum(old_sum, correction, block_sum)
        // --------------------------------------------------------

        for (int row = thread_id;
             row < valid_q;
             row += num_threads) {

            running_sum[row] =
                update_running_sum(
                    running_sum[row],
                    block_stat[row],
                    block_sum[row]
                );
        }

        __syncthreads();

        // --------------------------------------------------------
        // 2.11 累加 P tile @ V tile
        //
        // out_acc += p_tile @ v_tile
        //
        // 历史 out_acc 已经在前面乘过 correction。
        // --------------------------------------------------------

        accumulate_pv(
            p_tile,
            v_tile,
            out_acc,
            tile_q,
            tile_k,
            head_dim,
            thread_id,
            num_threads
        );

        // 下一轮将覆盖 k_tile、v_tile、p_tile，
        // 必须确保当前所有线程已经使用完成。
        __syncthreads();
    }

    // ============================================================
    // 3. 最终归一化并写回
    //
    // out = out_acc / running_sum
    // ============================================================

    const int valid_output_elements =
        valid_q * head_dim;

    for (int index = thread_id;
         index < valid_output_elements;
         index += num_threads) {

        const int row = index / head_dim;
        const int col = index % head_dim;

        out[(q_start + row) * head_dim + col] =
            out_acc[row * head_dim + col] /
            running_sum[row];
    }
}

# Step 24 - flash_attention_launcher (not yet solved)
# TODO: implement

# Step 25 - causal_mask (not yet solved)
# TODO: implement

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement

