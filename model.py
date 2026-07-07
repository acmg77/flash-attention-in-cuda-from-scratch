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

# Step 17 - load_tile (not yet solved)
# TODO: implement

# Step 18 - tile_scores (not yet solved)
# TODO: implement

# Step 19 - tile_rowmax (not yet solved)
# TODO: implement

# Step 20 - tile_exp (not yet solved)
# TODO: implement

# Step 21 - tile_rowsum (not yet solved)
# TODO: implement

# Step 22 - accumulate_pv (not yet solved)
# TODO: implement

# Step 23 - flash_attention_kernel (not yet solved)
# TODO: implement

# Step 24 - flash_attention_launcher (not yet solved)
# TODO: implement

# Step 25 - causal_mask (not yet solved)
# TODO: implement

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement

