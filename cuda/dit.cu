#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "common.h"
#include <torch/extension.h>

// intra-warp max-reduce
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// @layernorm
__global__ void layernorm_forward_kernel6(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, int N, int C, float eps) {
    // Layernorm without weight/bias
    assert(blockDim.x == WARP_SIZE);
    const bool save_mean = (mean != nullptr);
    const bool save_rstd = (rstd != nullptr);

    extern __shared__ char params[];
    x128* s_in = reinterpret_cast<x128*>(params) + (threadIdx.y * C / x128::size);

    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if(idx >= N) { return; } // guard

    // adjust pointers to current token
    inp += idx * C;
    out += idx * C;

    float sum = 0.0f;
    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = load128cs(inp + c);
        for(int k = 0; k < x128::size; ++k) {
            sum += (float)in_data[k];
        }
        s_in[c / x128::size] = in_data;
    }

    sum = warpReduceSum(sum);
    float m = sum / C;
    float v = 0.f;

    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = s_in[c / x128::size];
        for(int k = 0; k < x128::size; ++k) {
            v += ((float)in_data[k] - m) * ((float)in_data[k] - m);
        }
    }

    v = warpReduceSum(v) / C;
    float s = rsqrtf(v + eps);

    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = s_in[c / x128::size];
        x128 out_data;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * ((float)in_data[k] - m); // normalized output
            out_data[k] = n;
        }

        store128cs(out + c, out_data);
    }
    // cache the mean and rstd for the backward pass later
    if(threadIdx.x == 0 && save_mean) {
        __stcs(mean + idx, m);
    }
    // store the rstd, no need to cache it
    if(threadIdx.x == 0 && save_rstd) {
        __stcs(rstd + idx, s);
    }
}

// @layernorm
void launch_layernorm(float* out, float* mean, float* rstd,
                       const float* inp,
                       int B, int T, int C, float eps,
                       int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    int block_y = block_size / WARP_SIZE;
    const int grid_size = ceil_div(N, block_y);
    size_t smem = (2 + block_y) * C * sizeof(float);

    // in order to use more than 48 KiB of smem, need to call cudaFuncSetAttribute
    cudaCheck(cudaGetLastError());
    auto status = cudaFuncSetAttribute(layernorm_forward_kernel6, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (status == cudaSuccess) {
        layernorm_forward_kernel6<<<grid_size, dim3(WARP_SIZE, block_y), smem>>>(out, mean, rstd, inp, N, C, eps);
        cudaCheck(cudaGetLastError());
    } else {
        printf("Could not increase shared memory size to %zu Kbytes... Falling back to kernel 5!\n", smem/1000);
        exit(EXIT_FAILURE);
    }
    cudaCheck(cudaGetLastError());
}

// @partial_linear
namespace wt {
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                             float *As, float *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB) {
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    // float4 tmp;
    // asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
    //     : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
    //     : "l"(&A[(innerRowA + offset) * K + innerColA * 4]));
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
    // asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
    //     : "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 0]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 1]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 2]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 3])
    //     : "l"(&B[(innerRowB + offset) * N + innerColB * 4]));
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // populate registers for whole warptile
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }

    // execute warptile matmul
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // calculate per-thread results
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                regM[wSubRowIdx * TM + resIdxM] *
                regN[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}

} // namespace wt

/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMITER The number of subwarp tiling steps in M dimension.
 * @tparam WNITER The number of subwarp tiling steps in N dimension.
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 */

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
partial_linear_cuda(int M, int N, int K, int N_start, int N_end, float *A, float *B, float *C, float *bias) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;
  const uint N_out = N_end - N_start;

  // Placement of the warp in the threadblock tile
  const uint warpIdx = threadIdx.x / WARP_SIZE; // the warp this thread is in
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  // size of the warp subtile
  constexpr uint WMITER = (WM * WN) / (WARP_SIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER; // 64/2=32
  constexpr uint WSUBN = WN / WNITER; // 32/2=16

  // Placement of the thread in the warp subtile
  const uint threadIdxInWarp = threadIdx.x % WARP_SIZE;         // [0, 31]
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4

  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN + N_start;
  // Move C_ptr to warp's output tile
  C += (cRow * BM + warpRow * WM) * N_out + cCol * BN + warpCol * WN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
  // we cache into registers on the warptile level
  float regM[WMITER * TM] = {0.0f};
  float regN[WNITER * TN] = {0.0f};

  // Bias initialization, for each column, initialize threadResult with the bias

  // Move bias to N_start (partial offset) + current block column + current warp col within block
  bias += N_start + cCol * BN + warpCol * WN;

  for (int wSubColIdx=0; wSubColIdx < WNITER; wSubColIdx++){
    for (int tn=0; tn<TN; tn+=4){
      float4 bias_tn = reinterpret_cast<float4*>(&bias[
        wSubColIdx * WSUBN + /*warp sub col*/
        threadColInWarp * TN +
        tn
      ])[0];

      for (int wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++){
        for (int tm=0; tm<TM; tm++){
          const int idx = wSubRowIdx * TM * WNITER * TN +
                          tm * WNITER * TN + 
                          wSubColIdx * TN +
                          tn;
          threadResults[idx + 0] = bias_tn.x;
          threadResults[idx + 1] = bias_tn.y;
          threadResults[idx + 2] = bias_tn.z;
          threadResults[idx + 3] = bias_tn.w;
        }
      }
    }
  }

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();
    wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                        TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol,
                            threadRowInWarp, threadColInWarp);
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
    __syncthreads();
  }

  // write out the results
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // move C pointer to current warp subtile
      float *C_interim = C + (wSubRowIdx * WSUBM) * N_out + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          float4 tmp;
          // perform GEMM update in reg
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          tmp.x = threadResults[i + 0];
          tmp.y = threadResults[i + 1];
          tmp.z = threadResults[i + 2];
          tmp.w = threadResults[i + 3];
          // write back
          reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N_out +
                         threadColInWarp * TN + resIdxN])[0] = tmp;
        }
      }
    }
  }
}

// @partial_linear
void launch_partial_linear(int M, int N, int K, int N_start, int N_end, float *A, float *B, float *C, float* bias) {
    const int K10_NUM_THREADS = 128;
    const int K10_BN = 128;
    const int K10_BM = 128;
    const int K10_BK = 16;
    const int K10_WN = 64;
    const int K10_WM = 64;
    const int K10_WNITER = 4;
    const int K10_TN = 4;
    const int K10_TM = 8;
    dim3 blockDim(K10_NUM_THREADS);
  
    constexpr int NUM_WARPS = K10_NUM_THREADS / 32;
  
    // warptile in threadblocktile
    static_assert((K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0));
    static_assert((K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS);
  
    // threads in warpsubtile
    static_assert((K10_WM * K10_WN) % (WARP_SIZE * K10_TM * K10_TN * K10_WNITER) ==
                  0);
    constexpr int K10_WMITER =
        (K10_WM * K10_WN) / (32 * K10_TM * K10_TN * K10_WNITER);
    // warpsubtile in warptile
    static_assert((K10_WM % K10_WMITER == 0) and (K10_WN % K10_WNITER == 0));
  
    static_assert((K10_NUM_THREADS * 4) % K10_BK == 0,
                  "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                  "issues during GMEM->SMEM tiling (loading only parts of the "
                  "final row of Bs during each iteraion)");
    static_assert((K10_NUM_THREADS * 4) % K10_BN == 0,
                  "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                  "issues during GMEM->SMEM tiling (loading only parts of the "
                  "final row of As during each iteration)");
    static_assert(K10_BN % (16 * K10_TN) == 0,
                  "BN must be a multiple of 16*TN to avoid quantization effects");
    static_assert(K10_BM % (16 * K10_TM) == 0,
                  "BM must be a multiple of 16*TM to avoid quantization effects");
    static_assert((K10_BM * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                  "BM*BK must be a multiple of 4*256 to vectorize loads");
    static_assert((K10_BN * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                  "BN*BK must be a multiple of 4*256 to vectorize loads");
  
    const int N_out = N_end - N_start;
    dim3 gridDim(ceil_div(N_out, K10_BN), ceil_div(M, K10_BM));
    partial_linear_cuda<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
                        K10_TN, K10_NUM_THREADS>
        <<<gridDim, blockDim>>>(M, N, K, N_start, N_end, A, B, C, bias);
    cudaCheck(cudaGetLastError());
}
  
// @scale_and_shift
__global__ void scale_and_shift3(float* input, float* output, float* shift, float* scale, int B, int T, int C){
    const uint bt = blockIdx.y * 1  + (threadIdx.x / WARP_SIZE);
    const uint c = (blockIdx.x * 32 + (threadIdx.x % WARP_SIZE)) * 4;
    const uint b = bt / T;
    if (b >= B || bt >= B*T || c >= C) return;


    x128 scale_bc = load128cs(scale + b * C + c);
    x128 shift_bc = load128cs(shift + b * C + c);
    x128 inp_btc = load128cs(input + bt * C + c);

    x128 output_btc;
    #pragma unroll
    for (int k=0; k<x128::size; k++){
        output_btc[k] = inp_btc[k] * (1.0f + scale_bc[k]) + shift_bc[k];
    }
    store128cs(output + bt * C + c, output_btc);
}
// @scale_and_shift
void launch_scale_and_shift3(float* input, float* output, float* shift, float* scale, int B, int T, int C){
    dim3 gridDim_(ceil_div(C, 32*4), B*T);
    dim3 blockDim_(32);
    scale_and_shift3<<<gridDim_, blockDim_>>>(input, output, shift, scale, B, T, C);
    cudaCheck(cudaGetLastError());
}

// @rmsnorm
__global__ 
void rmsnorm_key_query_kernel2(float* QKV, float* query_weight, float* key_weight, float* rms_query, float* rms_key, float eps, int B, int T, int NH, int HS){
    // In-place version

    const bool save_rms = (rms_query != nullptr && rms_key != nullptr);
    const bool use_weight = (key_weight != nullptr && query_weight != nullptr);
    assert(blockDim.x == WARP_SIZE);

    // Load weights and inputs into shared memory
    extern __shared__ char params[];
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_in = reinterpret_cast<x128*>(params) + 2 * HS * (1 + threadIdx.y) / x128::size;

    const uint starting_idx = (threadIdx.y * WARP_SIZE + threadIdx.x) * x128::size;
    const uint stride = WARP_SIZE * blockDim.y * x128::size;
    for (int hs=starting_idx; hs<HS; hs+=stride){
        s_weight[hs/x128::size]        = load128cs(query_weight + hs);
        s_weight[(HS + hs)/x128::size] = load128cs(key_weight + hs);
    }
    __syncthreads();

    const uint bth = blockIdx.y * 32 + threadIdx.y; //[0, B*T*2*NH - 1]
    const uint h = bth % (2*NH);
    const uint bt = (bth-h) / (2*NH);
    assert(bth == bt * 2 * NH + h);
    const bool is_query = h < NH; /*First NH heads correpond to query, next NH heads correspond to key*/

    if (bth >= B*T*2*NH) return;
    // move pointers to current row, QKV (B, T, 3*NH*HS)
    QKV += bt * 3 * NH * HS + h * HS;

    // RMS computation (same code for query and key)
    float rms = 0.0f;
    for (int hs = threadIdx.x * x128::size; hs < HS; hs += WARP_SIZE * x128::size){
        x128 x = load128cs(QKV + hs);
        for (int k=0; k<x128::size; k++){
            rms += x[k] * x[k];
        }
        s_in[hs/x128::size] = x; /*store in SMEM for the rest of the code*/
    }
    rms = warpReduceSum(rms) / HS;
    float inv_rms = rsqrt(rms + eps);

    // Normalization
    for (int hs = threadIdx.x * x128::size; hs < HS; hs += WARP_SIZE * x128::size){
        x128 x = s_in[hs/x128::size];
        x128 w = (is_query) ? s_weight[hs/x128::size] : s_weight[(HS + hs)/x128::size];
        x128 y;
        for (int k=0; k<x128::size; ++k){
            float out_k = x[k] * inv_rms;
            if (use_weight) out_k *= w[k];
            y[k] = out_k;
        }
        store128cs(QKV + hs, y); //in-place
    }
    // save for backward
    if (save_rms && threadIdx.x == 0){
        if (is_query){
            rms_query[bt * NH + h] = inv_rms;
        } else{
            rms_key[bt * NH + (h - NH)] = inv_rms;
        }
    }
}
// @rms_norm
void launch_rmsnorm_query_key(float* QKV, float* query_weight, float* key_weight, float* rms_query, float* rms_key, float eps, int B, int T, int NH, int HS, bool verbose){

    // Increase SMEM size if possible
    int rows_per_block = 32;
    size_t SMEM_size = (1 + rows_per_block) * 2 * HS * sizeof(float);
    auto status = cudaFuncSetAttribute(rmsnorm_key_query_kernel2, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_size);
    // cudaCheck(cudaGetLastError());
    
    // If not possible, exit
    if (status != cudaSuccess){
        printf("Could not increase shared memory size to %zu Kbytes...\n", SMEM_size/1000);
        exit(EXIT_FAILURE);
    }
    // If possible, launch kernel
    // Each blocks treats "rows_per_block" rows, and there are B*T*NH*2 rows (corresponding to query and key, value is skipped) to be processed
    // for a total of B*T*NH*2 / "rows_per_block" blocks.
    // In each block, each warp of "WARP_SIZE" = 32 threads processes its own row of elements
    if (verbose) printf("Succesfully increased shared memory size to %zu Kbytes... Using kernel 5!\n", SMEM_size/1000);
    rmsnorm_key_query_kernel2
        <<<dim3(1, ceil_div(B*T*NH*2, rows_per_block)), dim3(WARP_SIZE, rows_per_block), SMEM_size>>>
        (QKV, query_weight, key_weight, rms_query, rms_key, eps, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}


// @fused_concat_permute
__global__ void fused_concat_permute(const float* c_qkv, const float* x_qkv, float* Q, float* K, float* V, int B, int Tc, int Tx, int C, int NH){
    // c_qkv (B, Tc, 3*NH, D), x_qkv (B, Tx, 3*NH, D)
    // Step 1: concat c_qkv, x_qkv -> (B, T = Tc + Tx, 3*NH, D)
    // Step 2: permute into 3 * (B, NH, T, D) where NH = num_heads

    const uint T = Tc + Tx;
    const uint D = C / NH;
    const uint bhtd = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const uint d = bhtd % D;
    const uint t = ((bhtd - d) / (D)) % T;
    const uint h = ((bhtd - d - t * D) / (T*D)) % NH;
    const uint b = bhtd / (NH * T * D);
    const bool read_from_context = t < Tc;

    assert(bhtd == b * NH * T * D + h * T * D + t * D + d);

    x128 inp_Q;
    x128 inp_K;
    x128 inp_V;
    if (read_from_context){
        inp_Q = load128cs(c_qkv + (b * Tc * 3 * C) + (t * 3 * C) + (h * D + d));
        inp_K = load128cs(c_qkv + (b * Tc * 3 * C) + (t * 3 * C) + (h * D + d + C));
        inp_V = load128cs(c_qkv + (b * Tc * 3 * C) + (t * 3 * C) + (h * D + d + 2*C));
    } else {
        // else read from x, replace Tc by Tx and adjust t
        inp_Q = load128cs(x_qkv + (b * Tx * 3 * C) + ((t - Tc) * 3 * C) + (h * D + d));
        inp_K = load128cs(x_qkv + (b * Tx * 3 * C) + ((t - Tc) * 3 * C) + (h * D + d + C));
        inp_V = load128cs(x_qkv + (b * Tx * 3 * C) + ((t - Tc) * 3 * C) + (h * D + d + 2*C));
    }

    store128cs(Q + bhtd, inp_Q);
    store128cs(K + bhtd, inp_K);
    store128cs(V + bhtd, inp_V);
}
// @fused_concat_permute
void launch_fused_concat_permute(const float* c_qkv, const float* x_qkv, float* Q, float* K, float* V, int B, int Tc, int Tx, int C, int NH){
    dim3 gridDim_(ceil_div(B * (Tc + Tx) * C, 1024*4));
    dim3 blockDim_(1024);
    fused_concat_permute<<<gridDim_, blockDim_>>>(c_qkv, x_qkv, Q, K, V, B, Tc, Tx, C, NH);
    cudaCheck(cudaGetLastError());
}

// @fused_unpermute_split
__global__
void fused_unpermute_split(const float* attn_out, float* c_out, float* x_out, int B, int Tc, int Tx, int C, int NH){
    // attn_out (B, NH, T, D) -> unpermute to (B, T, C=NH*D) -> split to (B, Tc, C) & (B, Tx, C)

    const uint T = Tc + Tx;
    const uint D = C / NH;
    const uint bhtd = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const uint d = bhtd % D;
    const uint t = ((bhtd - d) / (D)) % T; /*[0, Tc+Tx(*/
    const uint h = ((bhtd - d - t * D) / (T*D)) % NH;
    const uint b = bhtd / (NH * T * D);
    const bool load_to_context = t < Tc;

    x128 inp = load128cs(attn_out + bhtd);
    
    if (load_to_context){
        store128cs(c_out + (b * Tc * C) + (t * C) + (h * D + d), inp);
    } else {
        store128cs(x_out + (b * Tx * C) + ((t - Tc) * C) + (h * D + d), inp);
    }
}
// @fused_unpermute_split
void launch_fused_unpermute_split(const float* attn_out, float* c_out, float* x_out, int B, int Tc, int Tx, int C, int NH){
    dim3 gridDim_(ceil_div(B * (Tc + Tx) * C, 1024*4));
    dim3 blockDim_(1024);
    fused_unpermute_split<<<gridDim_, blockDim_>>>(attn_out, c_out, x_out, B, Tc, Tx, C, NH);
    cudaCheck(cudaGetLastError());
}

// @scale_and_softmax
__global__
void scale_softmax_kernel2(const float* input, float* output, int N, int T, float scale){
    const uint numWarps = ceil_div(blockDim.x, WARP_SIZE);
    const uint warpIdx = threadIdx.x / WARP_SIZE;
    const uint threadIdxWithinWarp = threadIdx.x % WARP_SIZE;
    const uint nt = blockIdx.x * numWarps + warpIdx;

    if (nt >= N * T) return;

    // Points to input[n,t,0]
    const float* x_nt = input + nt * T;
    float* y_nt = output + nt * T;

    // Online-softmax step1: compute max value and the sum of the exponentials in a single loop
    float maxval = -FLT_MAX;
    float old_maxval;
    float sumval = 0.0f;

    for (int l = threadIdxWithinWarp * x128::size; l < T; l += WARP_SIZE * x128::size){
        // read input
        x128 x_ntl = load128cs(x_nt + l);
        old_maxval = maxval;
        // update max
        for (int k=0; k<x128::size; k++){
            maxval = fmaxf(maxval, x_ntl[k]);
        }
        // update sum
        sumval *= expf(scale * (old_maxval - maxval));
        for (int k=0; k<x128::size; k++){
            sumval += expf(scale * (x_ntl[k] - maxval));
        }
    }
    // Intra-warp maximum reduce operation
    float global_maxval = warpReduceMax(maxval);
    // lane 0 (i.e. thread 0) of the warp gets the maximum value, other threads aren't guaranteed to have the same value
    // so we broadcast the maximum value to all threads
    global_maxval = __shfl_sync(0xffffffff, global_maxval, 0);

    // Re-adjust the sum
    sumval *= expf(scale * (maxval - global_maxval));
    // Intra-warp sum reduce operation then inversion
    float inv_denominator = 1.0f / warpReduceSum(sumval);

    // Online-softmax step2: Compute softmax
    // Divided into 2 parts, just in case T isn't a multiple of 4
    // First part: loop through T rounded down (not up) to the nearest multiple of 4
    int T_floor4 = (T / 4) * 4;
    for (int l = threadIdxWithinWarp * x128::size; l < T_floor4; l += WARP_SIZE * x128::size){
        // If there are at least 4 elements to be loaded, use vectorized load
        x128 x_ntl = load128cs(x_nt + l);
        x128 y_ntl;
        for (int k=0; k<x128::size; k++){
            y_ntl[k] = expf(scale * (x_ntl[k] - global_maxval)) * inv_denominator;
        }
        store128cs(y_nt + l, y_ntl);
    }
    // Second part: loop through remaining if needed
    for (int l=T_floor4; l<T; l++){
        y_nt[l] = expf(scale * (x_nt[l] - global_maxval)) * inv_denominator;
    }
}


// @attention
void mha_cuda(float* Q, float* K, float* V, float* preatt, float* att, float* attn_out,
                int B, int Tc, int Tx, int C, int NH) {
    // Q,K,V (B, NH, T, HS)
    const int T = Tc + Tx;
    const int HS = C / NH;

    // batched matrix multiply with cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                                     CUBLAS_OP_T, CUBLAS_OP_N,
                                     T, T, HS,
                                     &alpha,
                                     K, HS, T * HS,
                                     Q, HS, T * HS,
                                     &beta,
                                     preatt, T, T * T,
                                     B * NH));

    // multiply all elements of preatt elementwise by scale
    float scale = rsqrtf(HS);
    int softmax_block_size = 256;
    // round T to the next multiple of 4. Necessary only for softmax operation because of vectorized loads over the T axis
    int T_ = ceil_div(T, 4) * 4;
    int grid_size = ceil_div(B * NH * T_, 8);
    scale_softmax_kernel2<<<grid_size, softmax_block_size>>>(preatt, att, B*NH, T_, scale);

    // new approach: first cuBLAS another batched matmul
    // attn_out
    // attn_out = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     HS, T, T,
                                     &alpha,
                                     V, HS, T * HS,
                                     att, T, T * T,
                                     &beta,
                                     attn_out, HS, T * HS,
                                     B * NH));
}

// @fused_gate_add
__global__ 
void gate_multiply_residual_add(float* input, float* residual, float* output, float* gate, int B, int T, int C){
    // parallelizes over B*T*C
    const uint btc = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const uint b = btc / (T*C);
    const uint c = btc % C;
    if (btc >= B*T*C) return;

    x128 inp_btc = load128cs(input + btc);
    x128 res_btc = load128cs(residual + btc);
    x128 gate_bc = load128(gate + b * C + c);
    x128 output_btc;

    #pragma unroll
    for (int k=0; k<x128::size; k++){
        output_btc[k] = (inp_btc[k] * gate_bc[k]) + res_btc[k];
    }
    store128cs(output + btc, output_btc);
}
// @fused_gate_add
void launch_fused_gate_add(float* input, float* residual, float* output, float* gate, int B, int T, int C){
    dim3 gridDim_(ceil_div(B*T*C, 1024*4));
    dim3 blockDim_(1024);
    gate_multiply_residual_add<<<gridDim_, blockDim_>>>(input, residual, output, gate, B, T, C);
    cudaCheck(cudaGetLastError());
}

// @linear
void launch_linear(const float* inp, float* out, const float* weight, const float* bias,
            int B, int T, int C, int OC, bool has_gelu) {
    int has_bias = (bias != NULL);

    // check bias alignment
    if(((uintptr_t)bias % 16) != 0) {
        printf("Bias pointer is not aligned (cuBLASLt requirement)!\n");
        exit(EXIT_FAILURE);
    }

    int returnedResults = 0;
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatmulPreference_t preference;
    cublasLtMatrixLayout_t weightLayout;
    cublasLtMatrixLayout_t inputLayout;
    cublasLtMatrixLayout_t outputLayout;
    cublasLtMatrixLayout_t biasLayout;
    cublasLtMatmulHeuristicResult_t heuristic;

    // create the operation descriptor
    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opNoTranspose2 = CUBLAS_OP_N;
    cublasLtEpilogue_t epilogueBias = CUBLASLT_EPILOGUE_DEFAULT;
    if (has_bias && has_gelu) {
        epilogueBias = CUBLASLT_EPILOGUE_GELU_BIAS;
    } else if (has_bias) {
        epilogueBias = CUBLASLT_EPILOGUE_BIAS;
    } else if (has_gelu) {
        epilogueBias = CUBLASLT_EPILOGUE_GELU;
    }
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, cublas_compute_type, CUDA_R_32F));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opNoTranspose, sizeof(opNoTranspose)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opNoTranspose2, sizeof(opNoTranspose2)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogueBias, sizeof(epilogueBias)));
    if (has_bias) cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));

    // define matrix layouts
    cublasCheck(cublasLtMatrixLayoutCreate(&weightLayout, CUDA_R_32F, OC, C, OC));
    cublasCheck(cublasLtMatrixLayoutCreate(&inputLayout, CUDA_R_32F, C, B*T, C));
    cublasCheck(cublasLtMatrixLayoutCreate(&outputLayout, CUDA_R_32F, OC, B*T, OC));
    cublasCheck(cublasLtMatrixLayoutCreate(&biasLayout, CUDA_R_32F, OC, 1, OC));

    // create a preference handle with specified max workspace
    cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

    // find a suitable algorithm
    cublasCheck(cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc,
        weightLayout, inputLayout, outputLayout, outputLayout,
        preference, 1, &heuristic, &returnedResults));
    if (returnedResults == 0) {
        printf("No cuBLASLt algorithm: B: %d, T: %d, C: %d, OC: %d, bias: %d, gelu: %d\n",
            B, T, C, OC, has_bias, has_gelu);
        exit(EXIT_FAILURE);
    }

    // call the matmul:
    const float alpha = 1.0f, beta = 0.0f;
    cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc,
        &alpha, weight, weightLayout, inp, inputLayout, &beta,
        out, outputLayout, out, outputLayout, &heuristic.algo,
        cublaslt_workspace, cublaslt_workspace_size, 0));

    // cleanups
    cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
    cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
    cublasCheck(cublasLtMatrixLayoutDestroy(weightLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(inputLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(outputLayout));
    if (has_bias) cublasCheck(cublasLtMatrixLayoutDestroy(biasLayout));
    cudaCheck(cudaGetLastError());
}

// @mlp
void mlp_cuda(const float* input, float* hidden, float* output,
    const float* weight1, const float* bias1, const float* weight2, const float* bias2,
    int B, int T, int C_in, int C_hid, int C_out){
    
    launch_linear(input, hidden, weight1, bias1, B, T, C_in, C_hid, /*has_gelu=*/true);
    launch_linear(hidden, output, weight2, bias2, B, T, C_hid, C_out, /*has_gelu=*/false);
}

// @dit_block
void dit_block_cuda(
    // Inputs, outputs written in-place
    float* y, float* c, float* x,
    // Context parameters
    float* c_ada_lnorm_weights, float* c_ada_lnorm_biases, /*Adaptative layer norm*/
    float* c_Wqkv, float* c_bias_qkv, /*Attention*/
    float* c_rms_Wq, float* c_rms_Wk, /*rms norm weight*/
    float* c_Wout, float* c_bias_out, /*Attention*/
    float* c_mlp_W1, float* c_mlp_b1, float* c_mlp_W2, float* c_mlp_b2, /*mlp*/
    // Latent parameters
    float* x_ada_lnorm_weights, float* x_ada_lnorm_biases,
    float* x_Wqkv, float* x_bias_qkv, 
    float* x_rms_Wq, float* x_rms_Wk,
    float* x_Wout, float* x_bias_out,
    // Dual latent
    float* x_Wqkv_dual, float* x_bias_qkv_dual,
    float* x_rms_Wq_dual, float* x_rms_Wk_dual,
    float* x_Wout_dual, float* x_bias_out_dual,
    float* x_mlp_W1, float* x_mlp_b1, float* x_mlp_W2, float* x_mlp_b2, 
    // Intermediate activations
    float* c_hid, float* x_hid, float* x_hid_dual, float* c_hid2, float* x_hid2,
    float* c_qkv, float* x_qkv, float* Q, float* K, float* V,
    float* preattn, float* attn, float* attn_out,
    float* c_mlp_hidden, float* x_mlp_hidden,
    float* y_hid1, float* y_hid2,
    // Others
    int B, int Tc, int Tx, int emb_dim, int attn_heads, int mlp_expand, bool use_dual_attention, bool use_qk_norm, bool discard_context){

    
    // ***************************************************** SETUP ***********************************************************
    // Tx = number of tokens in latent vector (the image)
    // Tc = number of tokens in context embeddings (the text)
    // In the following, c (lower-case) will always refer to the context embeddings
    // and x refers to the latent vector
    const int T = Tx + Tc;
    const int head_dim = emb_dim / attn_heads;
    const int c_chunks = (discard_context) ? 2 : 6;
    const int x_chunks = (use_dual_attention) ? 9 : 6;

    // ***************************************************** MAIN PART ***********************************************************

    // *********************************** Pre-attention Context **********************************
    // c_hid = layernorm(c, weight=None, bias=None, save_for_backward=False)
    launch_layernorm(c_hid, nullptr, nullptr, c, B, Tc, emb_dim, 1e-6f, /*block_size=*/128);
    printf("lnorm c done\n");
    // scale and shift
    launch_partial_linear(B, c_chunks * emb_dim, emb_dim, /*start_index=*/0, /*end_index=*/emb_dim, y, c_ada_lnorm_weights, y_hid1, c_ada_lnorm_biases);
    printf("partial_linear c_shift done\n");
    launch_partial_linear(B, c_chunks * emb_dim, emb_dim, /*start_index=*/emb_dim, /*end_index=*/2*emb_dim, y, c_ada_lnorm_weights, y_hid2, c_ada_lnorm_biases);
    printf("partial_linear c_scale done\n");
    /*This is not strictly necessary, we only swap these to mimic the huggingface implementation*/
    if (discard_context){
        launch_scale_and_shift3(c_hid, c_hid, y_hid2, y_hid1, B, Tc, emb_dim);
    } else {
        launch_scale_and_shift3(c_hid, c_hid, y_hid1, y_hid2, B, Tc, emb_dim);
    }
    // Wqkv projections
    launch_linear(c_hid, c_qkv, c_Wqkv, c_bias_qkv, B, Tc, emb_dim, 3*emb_dim, /*has_gelu=*/false);
    // rms norm query and key
    if (use_qk_norm){
        launch_rmsnorm_query_key(c_qkv, c_rms_Wq, c_rms_Wk, nullptr, nullptr, 1e-6f, B, Tc, attn_heads, head_dim, false);
    }


    // *********************************** Pre-attention Latent **********************************
    launch_layernorm(x_hid, nullptr, nullptr, x, B, Tx, emb_dim, 1e-6f, /*block_size=*/128);

    if (use_dual_attention){
        launch_partial_linear(B, 9 * emb_dim, emb_dim, 6*emb_dim, 7*emb_dim, y, x_ada_lnorm_weights, y_hid1, x_ada_lnorm_biases);
        launch_partial_linear(B, 9 * emb_dim, emb_dim, 7*emb_dim, 8*emb_dim, y, x_ada_lnorm_weights, y_hid2, x_ada_lnorm_biases);
        launch_scale_and_shift3(x_hid, x_hid_dual, /*shift=*/y_hid1, /*scale=*/y_hid2, B, Tx, emb_dim);
    }

    launch_partial_linear(B, x_chunks * emb_dim, emb_dim, 0*emb_dim, 1*emb_dim, y, x_ada_lnorm_weights, y_hid1, x_ada_lnorm_biases);
    launch_partial_linear(B, x_chunks * emb_dim, emb_dim, 1*emb_dim, 2*emb_dim, y, x_ada_lnorm_weights, y_hid2, x_ada_lnorm_biases);
    launch_scale_and_shift3(x_hid, x_hid, /*shift=*/y_hid1, /*scale=*/y_hid2, B, Tx, emb_dim);
    
    launch_linear(x_hid, x_qkv, x_Wqkv, x_bias_qkv, B, Tx, emb_dim, 3*emb_dim, /*has_gelu=*/false);
    if (use_qk_norm){
        launch_rmsnorm_query_key(x_qkv, x_rms_Wq, x_rms_Wk, nullptr, nullptr, 1e-6f, B, Tx, attn_heads, head_dim, false);
    }

    // ************************************* Attention ********************************************
    // right now, x_qkv is (B, Tx, 3*NH, HS) and c_qkv is (B, Tc, 3*NH, HS)
    // we want to permute the token and head dimension, and concatenate c & x along the token dimension
    // then split it in 3 tensors: Q, K, V before performing self attention
    launch_fused_concat_permute(c_qkv, x_qkv, Q, K, V, B, Tc, Tx, emb_dim, attn_heads);
    // Self attention
    mha_cuda(Q, K, V, preattn, attn, attn_out, B, Tc, Tx, emb_dim, attn_heads);
    // attn_out is (B, NH, T, HS) -> unpermute to (B, T, NH, HS) re-interpreted as (B, T, C)
    //                            -> split to c_hid2 (B, Tc, C) and x_hid2 (B, Tx, C)
    launch_fused_unpermute_split(attn_out, c_hid2, x_hid2, B, Tc, Tx, emb_dim, attn_heads);
    // W_out projections
    if (!discard_context) {
        launch_linear(c_hid2, c_hid, c_Wout, c_bias_out, B, Tc, emb_dim, emb_dim, /*has_gelu=*/false);
    }
    launch_linear(x_hid2, x_hid, x_Wout, x_bias_out, B, Tx, emb_dim, emb_dim, /*has_gelu=*/false);
    
    // ************************************* Dual Attention ********************************************
    if (use_dual_attention){
        launch_linear(x_hid_dual, x_qkv, x_Wqkv_dual, x_bias_qkv_dual, B, Tx, emb_dim, 3*emb_dim, /*has_gelu=*/false);
        if (use_qk_norm) {
            launch_rmsnorm_query_key(x_qkv, x_rms_Wq_dual, x_rms_Wk_dual, nullptr, nullptr, 1e-6f, B, Tx, attn_heads, head_dim, false);
        }
        // re-using the temporary activations
        launch_fused_concat_permute(x_qkv, nullptr, Q, K, V, B, Tx, 0, emb_dim, attn_heads);
        mha_cuda(Q, K, V, preattn, attn, attn_out, B, Tx, 0, emb_dim, attn_heads);
        launch_fused_unpermute_split(attn_out, x_hid2, nullptr, B, Tx, 0, emb_dim, attn_heads);
        launch_linear(x_hid2, x_hid_dual, x_Wout_dual, x_bias_out_dual, B, Tx, emb_dim, emb_dim, /*has_gelu=*/false);
        // gating
        launch_partial_linear(B, 9 * emb_dim, emb_dim, 8*emb_dim, 9*emb_dim, y, x_ada_lnorm_weights, y_hid1, x_ada_lnorm_biases);
        launch_fused_gate_add(x_hid_dual, x, x, y_hid1, B, Tx, emb_dim);
    }

    // *********************************** Context MLP **********************************
    if (!discard_context){
        // gate
        launch_partial_linear(B, 6 * emb_dim, emb_dim, 2*emb_dim, 3*emb_dim, y, c_ada_lnorm_weights, y_hid1, c_ada_lnorm_biases);
        launch_fused_gate_add(c_hid, c, c, y_hid1, B, Tc, emb_dim);
        // Layer norm
        launch_layernorm(c_hid, nullptr, nullptr, c, B, Tc, emb_dim, 1e-6f, /*block_size=*/128);
        // scale and shift
        launch_partial_linear(B, 6 * emb_dim, emb_dim, /*start_index=*/3*emb_dim, /*end_index=*/4*emb_dim, y, c_ada_lnorm_weights, y_hid1, c_ada_lnorm_biases);
        launch_partial_linear(B, 6 * emb_dim, emb_dim, /*start_index=*/4*emb_dim, /*end_index=*/5*emb_dim, y, c_ada_lnorm_weights, y_hid2, c_ada_lnorm_biases);
        launch_scale_and_shift3(c_hid, c_hid, y_hid1, y_hid2, B, Tc, emb_dim);
        // MLP
        mlp_cuda(c_hid, c_mlp_hidden, c_hid2, c_mlp_W1, c_mlp_b1, c_mlp_W2, c_mlp_b2, B, Tc, emb_dim, mlp_expand*emb_dim, emb_dim);
        // gate
        launch_partial_linear(B, 6 * emb_dim, emb_dim, /*start_index=*/5*emb_dim, /*end_index=*/6*emb_dim, y, c_ada_lnorm_weights, y_hid1, c_ada_lnorm_biases);
        launch_fused_gate_add(c_hid2, c, c, y_hid1, B, Tc, emb_dim);
    }

    // *********************************** Latent MLP **********************************
    // gate
    launch_partial_linear(B, x_chunks * emb_dim, emb_dim, 2*emb_dim, 3*emb_dim, y, x_ada_lnorm_weights, y_hid1, x_ada_lnorm_biases);
    launch_fused_gate_add(x_hid, x, x, y_hid1, B, Tx, emb_dim);
    // GOOD UP UNTIL HERE
    // Layer norm increase error from 1e-3 to 1e-2
    launch_layernorm(x_hid, nullptr, nullptr, x, B, Tx, emb_dim, 1e-6f, /*block_size=*/128);

    // scale and shift
    launch_partial_linear(B, x_chunks * emb_dim, emb_dim, 3*emb_dim, 4*emb_dim, y, x_ada_lnorm_weights, y_hid1, x_ada_lnorm_biases);
    launch_partial_linear(B, x_chunks * emb_dim, emb_dim, 4*emb_dim, 5*emb_dim, y, x_ada_lnorm_weights, y_hid2, x_ada_lnorm_biases);
    launch_scale_and_shift3(x_hid, x_hid, y_hid1, y_hid2, B, Tx, emb_dim);
    // MLP
    mlp_cuda(x_hid, x_mlp_hidden, x_hid2, x_mlp_W1, x_mlp_b1, x_mlp_W2, x_mlp_b2, B, Tx, emb_dim, mlp_expand*emb_dim, emb_dim);
    // gate
    launch_partial_linear(B, x_chunks * emb_dim, emb_dim, 5*emb_dim, 6*emb_dim, y, x_ada_lnorm_weights, y_hid1, x_ada_lnorm_biases);
    launch_fused_gate_add(x_hid2, x, x, y_hid1, B, Tx, emb_dim);
}

std::vector<torch::Tensor> DiT_block_forward_cuda(
    // Inputs
    torch::Tensor& temp_embeddings, torch::Tensor& context_embeddings, torch::Tensor& latent_vector,
    // Params
    std::unordered_map<std::string, torch::Tensor>& params,
    // Others
    int B, int Tc, int Tx, int emb_dim, int attn_heads, int mlp_expand, bool use_dual_attention, bool use_qk_norm, bool discard_context){
    
    // ***************************************************** SETUP ***********************************************************
    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    cublas_compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
    cublasMath_t cublas_math_mode = CUBLAS_TF32_TENSOR_OP_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
    // setup the (global) cuBLASLt workspace
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));


    // In the following, c (lower-case) will always refer to the context embeddings
    // and x refers to the latent vector
    int T = Tc + Tx;
    // Pointers
    float* y = temp_embeddings.data_ptr<float>();
    float* c = context_embeddings.data_ptr<float>();
    float* x = latent_vector.data_ptr<float>();

    // utility function to retrieve the data ptr of a parameter by its name. If the name doesn't exist, returns nullptr
    auto get_ptr = [&params](const std::string& param_name) -> float* {
        if (params.count(param_name) > 0) {
            return params[param_name].data_ptr<float>();
        }
        // std::cout << "Skipping missing param: " << param_name << std::endl;
        return nullptr;
    };

    // Intermediate activation pointers
    float* x_hid_dual = nullptr;
    if (use_dual_attention){
        cudaCheck(cudaMalloc(&x_hid_dual, B*Tx*emb_dim*sizeof(float)));
    }
    // attention
    float* c_qkv;
    float* x_qkv;
    float* Q;
    float* K;
    float* V;
    float* preattn;
    float* attn;
    float* attn_out;
    cudaCheck(cudaMalloc(&c_qkv, B * Tc * 3 * emb_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&x_qkv, B * Tx * 3 * emb_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&Q,   B * T * emb_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&K,   B * T * emb_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&V,   B * T * emb_dim * sizeof(float)));
    // round T to the next multiple of 4. Necessary only for softmax operation because of vectorized loads over the T axis
    int T_ = ceil_div(T, 4) * 4;
    cudaCheck(cudaMalloc(&preattn, B * attn_heads * T_ * T_ * sizeof(float)));
    cudaCheck(cudaMalloc(&attn, B * attn_heads * T_ * T_ * sizeof(float)));
    cudaCheck(cudaMalloc(&attn_out, B * T * emb_dim * sizeof(float)));
    // Others
    float* c_hid;
    float* x_hid;
    float* c_hid2;
    float* x_hid2;
    float* y_hid1;
    float* y_hid2;
    float* c_mlp_hid;
    float* x_mlp_hid;
    cudaCheck(cudaMalloc(&c_hid,     B * Tc * emb_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&x_hid,     B * Tx * emb_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&c_hid2,    B * Tc * emb_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&x_hid2,    B * Tx * emb_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&y_hid1,    B * emb_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&y_hid2,    B * emb_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&c_mlp_hid, B * Tc * mlp_expand * emb_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&x_mlp_hid, B * Tx * mlp_expand* emb_dim * sizeof(float)));

    // torch::Tensor toto = torch::empty({B, Tx, emb_dim}, torch::TensorOptions().device(torch::kCUDA));
    // torch::Tensor tata = torch::empty({B, Tx, emb_dim}, torch::TensorOptions().device(torch::kCUDA));

    printf("Shape check:\n");
    std::cout << "y : " << temp_embeddings.sizes() << std::endl;
    std::cout << "c : " << context_embeddings.sizes() << std::endl;
    std::cout << "x : " << latent_vector.sizes() << std::endl;
    for (auto& it: params){
        std::cout << it.first << " : " << it.second.sizes() << std::endl;
    }

    dit_block_cuda(
        y, c, x,
        // Context
        get_ptr("context_ada_lnorm.linear.weight"), get_ptr("context_ada_lnorm.linear.bias"),
        get_ptr("context_to_qkv.weight"), get_ptr("context_to_qkv.bias"),
        get_ptr("context_rmsnorm_query.weight"), get_ptr("context_rmsnorm_key.weight"),
        get_ptr("context_attn_Wout.weight"), get_ptr("context_attn_Wout.bias"),
        get_ptr("context_mlp.lin1.weight"), get_ptr("context_mlp.lin1.bias"),
        get_ptr("context_mlp.lin2.weight"), get_ptr("context_mlp.lin2.bias"),
        // Latent parameters
        get_ptr("latent_ada_lnorm.linear.weight"), get_ptr("latent_ada_lnorm.linear.bias"),
        get_ptr("latent_to_qkv.weight"), get_ptr("latent_to_qkv.bias"),
        get_ptr("latent_rmsnorm_query.weight"), get_ptr("latent_rmsnorm_key.weight"),
        get_ptr("latent_attn_Wout.weight"), get_ptr("latent_attn_Wout.bias"),
        // Dual latent parameters
        get_ptr("latent_dual_to_qkv.weight"), get_ptr("latent_dual_to_qkv.bias"),
        get_ptr("latent_dual_rmsnorm_query.weight"), get_ptr("latent_dual_rmsnorm_key.weight"),
        get_ptr("latent_dual_attn_Wout.weight"), get_ptr("latent_dual_attn_Wout.bias"),
        // Latent MLP
        get_ptr("latent_mlp.lin1.weight"), get_ptr("latent_mlp.lin1.bias"),
        get_ptr("latent_mlp.lin2.weight"), get_ptr("latent_mlp.lin2.bias"),
        // Intermediate activations
        c_hid, x_hid, x_hid_dual, c_hid2, x_hid2,
        c_qkv, x_qkv, Q, K, V,
        preattn, attn, attn_out,
        c_mlp_hid, x_mlp_hid,
        y_hid1, y_hid2,
        // Others
        B, Tc, Tx, emb_dim, attn_heads, mlp_expand, use_dual_attention, use_qk_norm, discard_context
    );
    cudaCheck(cudaGetLastError());

    // Free memory
    if (use_dual_attention) cudaCheck(cudaFree(x_hid_dual));
    cudaCheck(cudaFree(c_qkv));
    cudaCheck(cudaFree(x_qkv));
    cudaCheck(cudaFree(Q));
    cudaCheck(cudaFree(K));
    cudaCheck(cudaFree(V));
    cudaCheck(cudaFree(preattn));
    cudaCheck(cudaFree(attn));
    cudaCheck(cudaFree(attn_out));
    cudaCheck(cudaFree(x_hid));
    cudaCheck(cudaFree(c_hid));
    cudaCheck(cudaFree(x_hid2));
    cudaCheck(cudaFree(c_hid2));
    cudaCheck(cudaFree(y_hid1));
    cudaCheck(cudaFree(y_hid2));
    cudaCheck(cudaFree(c_mlp_hid));
    cudaCheck(cudaFree(x_mlp_hid));
    // return c, x
    return {context_embeddings, latent_vector};
}