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

# Step 10 - softmax_rows (not yet solved)
# TODO: implement

# Step 11 - pv_matmul (not yet solved)
# TODO: implement

# Step 12 - naive_attention (not yet solved)
# TODO: implement

# Step 13 - online_max (not yet solved)
# TODO: implement

# Step 14 - correction_factor (not yet solved)
# TODO: implement

# Step 15 - update_running_sum (not yet solved)
# TODO: implement

# Step 16 - rescale_output (not yet solved)
# TODO: implement

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

