#include <iostream>
#include <algorithm>
#include <functional>
#include <torch/extension.h>
#include <cassert>
#include <stdexcept>

void layernorm(float* x, float* y, float* mean, float* rstd, float* w, float* bias, float eps, int B, int T, int C){

    const bool use_weight = (w != nullptr);
    const bool use_bias = (bias != nullptr);
    const bool save_mean = (mean != nullptr);
    const bool save_rstd = (rstd != nullptr);

    for (int b=0; b<B; b++){
        for (int t=0; t<T; t++){
            // Pointer to x[b][t]
            float* x_bt = x + b*T*C + t*C;
            
            // mean
            float mean_ = 0.0f;
            for (int c=0; c<C; c++){
                mean_ += x_bt[c];
            }
            mean_ /= C;
            if (save_mean) mean[b*T + t] = mean_; /*Save for backward*/
            
            // inverse std
            float rstd_ = 0.0f;
            for (int c=0; c<C; c++){
                float shift_ = x_bt[c] - mean_;
                rstd_ += shift_ * shift_;
            }
            rstd_ /= C;
            rstd_ = 1.0f / sqrtf(rstd_+eps);
            if (save_rstd) rstd[b*T + t] = rstd_; /*Save for backward*/
            
            /*pointer to y[b][t]*/
            float* y_bt = y + b*T*C + t*C; 
            for (int c = 0; c < C; c++) {
                y_bt[c] = (rstd_ * (x_bt[c] - mean_)); // normalized input
                if (use_weight) y_bt[c] *= w[c];
                if (use_bias) y_bt[c] += bias[c];
            }
        }
    }
}

std::vector<torch::Tensor> LayerNorm_forward(
    torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor bias,
    float eps = 1e-5,
    bool save_for_backward = true
    ){
    // ************************************************** SETUP *******************************************************
    int B = input.size(0), T = input.size(1), C = input.size(2);
    // Allocate memory for output tensors
    float* μ = nullptr;
    float* σ = nullptr;
    torch::Tensor mean;
    torch::Tensor rstd;
    if (save_for_backward){
        mean = torch::empty({B, T});
        rstd = torch::empty({B, T});
        μ = mean.data_ptr<float>();
        σ = rstd.data_ptr<float>();
    }

    torch::Tensor output = torch::empty_like(input);
    float* x = input.data_ptr<float>();
    float* y = output.data_ptr<float>();
    float* w = weight.data_ptr<float>();
    float* b = bias.data_ptr<float>();

    // ************************************************** MAIN PART OF THE CODE *******************************************************
    
    layernorm(x, y, μ, σ, w, b, eps, B, T, C);

    if (save_for_backward) return {output, mean, rstd};
    return {output};
}


std::vector<torch::Tensor> LayerNorm_backward(
    torch::Tensor& output_grad,
    torch::Tensor& forward_input,
    torch::Tensor& weight,
    torch::Tensor& mean,
    torch::Tensor& rstd
){

    // ********************************** SETUP *************************************
    int B = forward_input.size(0), T = forward_input.size(1), C = forward_input.size(2);
    // Allocate memory and initialize gradients
    torch::Tensor input_grad = torch::zeros_like(forward_input);
    torch::Tensor weight_grad = torch::zeros_like(weight);
    torch::Tensor bias_grad = torch::zeros_like(weight);
    // Access data pointers
    float* dout = output_grad.data_ptr<float>();
    float* x = forward_input.data_ptr<float>();
    float* γ = weight.data_ptr<float>();
    float* μ = mean.data_ptr<float>();
    float* σ = rstd.data_ptr<float>();
    float* dx = input_grad.data_ptr<float>();
    float* dw = weight_grad.data_ptr<float>();
    float* db = bias_grad.data_ptr<float>();
    
    // ************************** MAIN PART OF THE CODE *****************************

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            float* x_bt = x + b * T * C + t * C;
            float* dx_bt = dx + b * T * C + t * C;
            float mean_bt = μ[b * T + t];
            float rstd_bt = σ[b * T + t];

            // first: two reduce operations
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int c = 0; c < C; c++) {
                float norm_btc = (x_bt[c] - mean_bt) * rstd_bt;
                float dnorm_btc = γ[c] * dout_bt[c];
                dnorm_mean += dnorm_btc;
                dnorm_norm_mean += dnorm_btc * norm_btc;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            for (int c = 0; c < C; c++) {
                float norm_btc = (x_bt[c] - mean_bt) * rstd_bt;
                float dnorm_btc = γ[c] * dout_bt[c];
                // gradient contribution to bias
                db[c] += dout_bt[c];
                // gradient contribution to weight
                dw[c] += norm_btc * dout_bt[c];
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_btc; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_btc * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dx_bt[c] += dval;
            }
        }
    }
    return {input_grad, weight_grad, bias_grad};
}

void rmsnorm(float* x, float* y, float* w, float* rms, float eps, int B, int T, int C){
 
    const bool use_weight = (w != nullptr);
    const bool save_rms = (rms != nullptr);
    const bool in_place = (x == y);

    for (int b=0; b<B; b++){
        for (int t=0; t<T; t++){
            // Pointer to x[b][t]
            float* x_bt = x + b*T*C + t*C;

            // Compute rms
            float rms_bt = 0.0f;
            for (int c=0; c<C; c++){
                rms_bt += x_bt[c] * x_bt[c];
            }
            rms_bt /= C;
            rms_bt = 1.0 / sqrtf(eps + rms_bt);
            if (save_rms) rms[b*T + t] = rms_bt;
            
            if (in_place){
                for (int c = 0; c < C; c++) {
                    x_bt[c] *= rms_bt;
                    if(use_weight) x_bt[c] *= w[c];
                }
            } else {
                /*pointer to y[b][t]*/
                float* y_bt = y + b*T*C + t*C;
                for (int c = 0; c < C; c++) {
                    y_bt[c] = x_bt[c] * rms_bt;
                    if(use_weight) y_bt[c] *= w[c];
                }
            }
        }
    }
}


void rmsnorm_group(float* x, float* y, float* w, float eps, int groups, int B, int T, int C){
    // rmsnorm per group of channels: C is divided into 'groups' groups. 
    // Equivalent to viewing x as (B,T,groups,C/groups) and rmsnorming over C/groups
    // w is of size C/groups (which means the weights are shared across groups)
    const int C_group = C / groups;
    const bool use_weight = (w != nullptr);

    for (int b=0; b<B; b++){
        for (int t=0; t<T; t++){
            // Pointer to x[b][t][g]
            float* x_btg = x + b*T*C + t*C;
            // Compute rms per group
            for (int g=0; g<groups; g++){
                float rms_bt = 0.0f;
                for (int c=0; c<C_group; c++){
                    rms_bt += x_btg[c] * x_btg[c];
                }
                rms_bt /= C_group;
                rms_bt = 1.0 / sqrtf(eps + rms_bt);

                for (int c = 0; c < C_group; c++) {
                    x_btg[c] *= rms_bt;
                    if(use_weight) x_btg[c] *= w[c];
                }

                x_btg += C_group;
            }
        }
    }
}

std::vector<torch::Tensor> RMSNorm_forward(
    torch::Tensor& input,
    torch::Tensor& weight,
    float eps,
    bool save_for_backward
    ){
    // ******************************** SETUP ***************************************
    int B = input.size(0), T = input.size(1), C = input.size(2);

    // If save_for_backward, save RMS statistics for backward pass later otherwise rms = nullptr
    float* rms = nullptr;
    torch::Tensor RMS;
    if (save_for_backward){
        RMS = torch::empty({B, T});
        rms = RMS.data_ptr<float>();
    }
    
    torch::Tensor output = torch::empty_like(input);
    float* x = input.data_ptr<float>();
    float* y = output.data_ptr<float>();
    float* w = weight.data_ptr<float>();


    // ******************************* MAIN PART OF THE CODE *******************************
    rmsnorm(x, y, w, rms, eps, B, T, C);

    if (save_for_backward) return {output, RMS};
    return {output};
}


std::vector<torch::Tensor> RMSNorm_backward(
    torch::Tensor& output_grad,
    torch::Tensor& forward_input,
    torch::Tensor& RMS,
    torch::Tensor& weight
    ){
    // ******************************** SETUP ***************************************
    int B = output_grad.size(0), T = output_grad.size(1), C = output_grad.size(2);
    // Allocate memory for output tensors
    torch::Tensor input_grad = torch::zeros_like(output_grad);
    torch::Tensor weight_grad = torch::zeros_like(weight);
    // Access the tensor's data pointers
    float* dx = input_grad.data_ptr<float>();
    float* dw = weight_grad.data_ptr<float>();
    float* dout = output_grad.data_ptr<float>();
    float* x = forward_input.data_ptr<float>();
    float* rms = RMS.data_ptr<float>();
    float* w = weight.data_ptr<float>();

    // ******************************* MAIN PART OF THE CODE *******************************
    for (int b=0; b<B; b++){
        for (int t=0; t<T; t++){
            float* x_bt = x + b*T*C + t*C;
            float* dx_bt = dx + b*T*C + t*C;
            float* dout_bt = dout + b*T*C + t*C;
            float rms_bt = rms[b*T + t];

            // mean along C
            float mean_bt = 0.0f;
            for (int c=0; c<C; c++){
                mean_bt += dout_bt[c] * x_bt[c] * w[c];
            }
            mean_bt /= C;
            
            for (int c=0; c<C; c++){
                float x_norm_btc = x_bt[c] * rms_bt;
                dw[c] += dout_bt[c] * x_norm_btc;
                dx_bt[c] += (dout_bt[c] * w[c] - ((x_norm_btc * mean_bt) * rms_bt) ) * rms_bt;
            }
        }
    }

    return {input_grad, weight_grad};
}

void gelu(float* x, float* y, long long n_elem){
    float sqrt2 = sqrtf(2.0);    
    for (int i=0; i<n_elem; i++){
        y[i] = 0.5 * x[i] * (1.0 + erff(x[i] / sqrt2));
    }
}

void gelu_tanh(float* input, float* output, long long n_elem){
    constexpr float c = 0.044715f;
    const float p = sqrtf(2.0 * M_1_PIf32);
    for (int i=0; i<n_elem; i++){
        float x = input[i];
        output[i] = 0.5 * x * (1.0 + tanhf(p * (x + c*x*x*x)) );
    }
}

torch::Tensor GELU_forward(torch::Tensor& input){

    // ********************************** SETUP *************************************
    // Output tensor
    torch::Tensor output = torch::empty_like(input);
    // Create pointers
    float* x = input.data_ptr<float>();
    float* y = output.data_ptr<float>();
    

    // ************************** MAIN PART OF THE CODE *****************************
    long long n_elem = input.numel();
    gelu(x, y, n_elem);
    return output;
}


torch::Tensor GELU_backward(torch::Tensor& grad_output, torch::Tensor& forward_input){

    // ********************************** SETUP *************************************
    // Create gradient tensor
    torch::Tensor input_grad = torch::empty_like(forward_input);
    // Pointers
    float* dout = grad_output.data_ptr<float>();
    float* x = forward_input.data_ptr<float>();
    float* dx = input_grad.data_ptr<float>();

    // ************************** MAIN PART OF THE CODE *****************************
    const float pi = 3.141592653589793;
    float sqrt2 = sqrtf(2.0);
    float inv_sqrt2pi = 1.0/sqrtf(2.0 * pi);
    long long n_elem = input_grad.numel();
    float phi_x, x_;

    // GELU'(x) = 0.5 * (1 + erf(x/sqrt(2))) + x * phi(x)
    // where phi(x) = 1/sqrt(2π) exp(-x**2/2) gaussian density
    for (int i=0; i<n_elem; i++){
        x_ = x[i];
        phi_x = inv_sqrt2pi * expf(-0.5 * x_ * x_);
        dx[i] = dout[i] * (0.5 * (1.0 + erff(x_/sqrt2)) + x_ * phi_x);
    }
    return input_grad;
}

void silu(float* x, float* y, long long n_elem){
    // SiLU(x) = x * sigmoid(x)
    for (int i=0; i<n_elem; i++){
        float sigmoid_x = 1.0 / (1.0 + expf(-x[i]));
        y[i] = x[i] * sigmoid_x;
    }
}

torch::Tensor SiLU_forward(torch::Tensor& input){

    // ********************************** SETUP *************************************
    // Output tensor
    torch::Tensor output = torch::empty_like(input);
    // Create pointers
    float* x = input.data_ptr<float>();
    float* y = output.data_ptr<float>();
    

    // ************************** MAIN PART OF THE CODE *****************************
    long long n_elem = input.numel();
    silu(x, y, n_elem);
    return output;
}


torch::Tensor SiLU_backward(torch::Tensor& grad_output, torch::Tensor& forward_input){

    // ********************************** SETUP *************************************
    // Create gradient tensor
    torch::Tensor input_grad = torch::empty_like(forward_input);
    // Pointers
    float* dout = grad_output.data_ptr<float>();
    float* x = forward_input.data_ptr<float>();
    float* dx = input_grad.data_ptr<float>();

    // ************************** MAIN PART OF THE CODE *****************************
    // SiLU'(x) = sigmoid(x) + x * sigmoid * (1 - sigmoid)
    // sigmoid'(x) = 1.0/(1.0 + exp(-x))**2 * exp(-x) = exp(-x) / (1.0 + exp(-x))**2
    long long n_elem = input_grad.numel();
    for (int i=0; i<n_elem; i++){
        float sigmoid_x = 1.0 / (1.0 + expf(-x[i]));
        dx[i] = dout[i] * (sigmoid_x * (1.0 + x[i] - (x[i] * sigmoid_x))); 
    }
    return input_grad;
}



void matmul(float* A, float* B, float* result, int M, int N, int P){
    // Regular matrix multiplication A @ B, where A is of size M x N, B of size N x P

    for (int m=0; m<M; m++){
        for (int p=0; p<P; p++){
            int row_A = m * N; /*row offset for A*/
            float dot_product_accum = 0.0f;
            // \sum_n A[m][n] * B[n][p]
            for (int n=0; n<N; n++){
                dot_product_accum += A[row_A + n] * B[n * P + p];
            }
            // write result at result[m][p]
            result[m * P + p] = dot_product_accum;
        }
    }
}


void matmul_Btranspose(float* A, float* B, float* result, int M, int N, int P){
    // matrix multiplication A @ B.T, where A is of size (M,N), B of size (P,N)

    for (int m=0; m<M; m++){
        for (int p=0; p<P; p++){
            int row_A = m * N; /*row offset for A*/
            float dot_product_accum = 0.0f;
            // \sum_n A[m][n] * B.T[n][p]
            for (int n=0; n<N; n++){
                dot_product_accum += A[row_A + n] * B[p * N + n];
            }
            // write result at result[m][p]
            result[m * P + p] = dot_product_accum;
        }
    }
}


void matmul_Atranspose(float* A, float* B, float* result, int M, int N, int P){
    // matrix multiplication result = A.T @ B, where A is of size (N,M), B of size (N,P), result of size (M,P)

    for (int m=0; m<M; m++){
        for (int p=0; p<P; p++){
            float dot_product_accum = 0.0f;
            // \sum_n A.T[m][n] * B[n][p]
            for (int n=0; n<N; n++){
                dot_product_accum += A[n * M + m] * B[n * P + p];
            }
            // write result at result[m][p]
            result[m * P + p] = dot_product_accum;
        }
    }
}


void bmm(float* A, float* V, float* result, int B, int T, int L, int C){
    // batched-matrix multiplication A @ K with A (B, T, L), V (B, L, C) -> (B,T,C)

    for (int b=0; b<B; b++){
        float* V_b = V + b * L * C;
        for (int t=0; t<T; t++){
            float* A_bt = A + b * T * L + t * L;
            float* res_bt = result + b * T * C + t * C;
            for (int c=0; c<C; c++){
                // \sum_l A[b,t,l] * V[b,l,c]
                float dot_product_accum = 0.0f;
                for (int l=0; l<L; l++){
                    dot_product_accum += A_bt[l] * V_b[l * C + c];
                }
                // write into result[b,t,c]
                res_bt[c] = dot_product_accum;
            }
        }
    }
}


void linear(float* x, float* W, float* b, float* y, int B, int C_in, int C_out){
    // Perform y = x @ W + b
    for (int i=0; i<B; i++){
        for (int k=0; k<C_out; k++){
            float dot_product_accum = 0.0f;
            // row offsets for x:
            int row_x = i * C_in;
            for (int j=0; j<C_in; j++){
                // dot_product_accum += x[i][j] * W[j][k];
                dot_product_accum += x[row_x + j] * W[j * C_out + k];
            }
            // write result at y[i][k]
            y[i * C_out + k] = dot_product_accum + b[k];
        }
    }
}

void to_qkv(float* x, float* Wqkv, float* bias_qkv, float* query, float* key, float* value, int B, int C_in, int C_out){
    // x @ Wqkv + bias (B, 3*C_out) -> split into query key value
    for (int m=0; m<B; m++){

        // Pointers
        float* x_m = x + m * C_in;
        float* Wk = Wqkv + C_out; /*Points to Wqkv[0, C_out], the next C_out elements correspond to the weights of Wk[0,:]*/
        float* Wv = Wqkv + 2 * C_out; /*Points to Wqkv[0, 2*C_out], the next C_out elements correspond to the weights of Wv[0,:]*/

        for (int n=0; n<C_out; n++){
            float dot_product_accum_query = bias_qkv[n];
            float dot_product_accum_key = bias_qkv[C_out + n];
            float dot_product_accum_value = bias_qkv[2 * C_out + n];

            for (int k=0; k<C_in; k++){
                const float x_mk = x_m[k];
                // += x[m,k] * W[k,n]
                dot_product_accum_query += x_mk * Wqkv[k * 3 * C_out + n];
                // += x[m,k] * W[k,C_out + n]
                dot_product_accum_key += x_mk * Wk[k * 3 * C_out + n];
                // += x[m,k] * W[k,2*C_out + n]
                dot_product_accum_value += x_mk * Wv[k * 3 * C_out + n];
            }
            
            // Write results
            query[m * C_out + n] = dot_product_accum_query;
            key[m * C_out + n] = dot_product_accum_key;
            value[m * C_out + n] = dot_product_accum_value;
        }

    }
}

void add(float* input, float* other, float* output, long long size){
    // element-wise add
    for (int i=0; i<size; i++){
        output[i] = input[i] + other[i];
    }
}

void mul(float* input, float* other, float* output, long long size){
    // element-wise multiplication
    for (int i=0; i<size; i++){
        output[i] = input[i] * other[i];
    }
}

torch::Tensor Linear_forward(torch::Tensor& input, torch::Tensor& weight, torch::Tensor& bias){
    // ********************************** SETUP *************************************
    int C_in = weight.size(0), C_out = weight.size(1);
    auto input_shape = input.sizes();
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end()-1);
    int batch_size = 1;
    for (auto dim: output_shape){
        batch_size *= dim;
    }
    output_shape.push_back(C_out);
    // Output tensor
    torch::Tensor output = torch::empty(output_shape);
    // Pointers
    float* x = input.data_ptr<float>();
    float* W = weight.data_ptr<float>();
    float* b = bias.data_ptr<float>();
    float* y = output.data_ptr<float>();


    // ************************** MAIN PART OF THE CODE *****************************
    linear(x, W, b, y, batch_size, C_in, C_out);
    return output;
}

std::vector<torch::Tensor> Linear_backward(
    torch::Tensor& output_grad, 
    torch::Tensor& forward_input, 
    torch::Tensor& weight){

    // ********************************** SETUP *************************************
    int batch_size = 1;
    for (int i=0; i<forward_input.dim()-1; i++){
        batch_size *= forward_input.size(i);
    }
    int C_in = weight.size(0), C_out = weight.size(1);
    // Output gradient tensors
    torch::Tensor input_grad = torch::empty_like(forward_input);
    torch::Tensor weight_grad = torch::empty_like(weight);
    torch::Tensor bias_grad = torch::empty({C_out});
    // Access data pointers
    float* dout = output_grad.data_ptr<float>();
    float* x = forward_input.data_ptr<float>();
    float* W = weight.data_ptr<float>();
    float* dx = input_grad.data_ptr<float>();
    float* dW = weight_grad.data_ptr<float>();
    float* db = bias_grad.data_ptr<float>();
    
    // ************************** MAIN PART OF THE CODE *****************************
    // dW = x.T @ dout  --- (C_in, B) @ (B, C_out) -> (C_in, C_out)
    matmul_Atranspose(x, dout, dW, C_in, batch_size, C_out);

    // dx = dout @ W.T --- (B, C_out) @ (C_out, C_in) -> (B, C_in)
    matmul_Btranspose(dout, W, dx, batch_size, C_out, C_in);
    
    // db = dout.sum(0) --- (B, C_out) -> (C_out)
    for (int c_out=0; c_out<C_out; c_out++){
        float accum_sum = 0.0f;
        for (int b=0; b<batch_size; b++){
            // accum_sum += dout[i][j];
            accum_sum += dout[b * C_out + c_out];
        }
        db[c_out] = accum_sum;
    }

    return {input_grad, weight_grad, bias_grad};
}

void softmax(float* x, float* y, int dim, int batch_size){
    // softmax on last axis

    for (int i=0; i<batch_size; i++){
        float* x_i = x + i * dim;
        float* y_i = y + i * dim;
        // Compute the max
        float max_ = -9999.9f;
        for (int j=0; j<dim; j++){
            if (x_i[j] > max_) max_ = x_i[j];
        }
        
        float sum_exp = 0.0f;
        for (int j=0; j<dim; j++){
            float exp_ = expf(x_i[j] - max_);
            y_i[j] = exp_;
            sum_exp += exp_;
        }

        for (int j=0; j<dim; j++){
            y_i[j] /= sum_exp;
        }
    }
}


void softmax_inplace(float* x, int dim, int batch_size){
    // In-place softmax on last axis

    for (int i=0; i<batch_size; i++){
        float* x_i = x + i * dim;
        // Compute the max
        float max_ = -9999.9f;
        for (int j=0; j<dim; j++){
            if (x_i[j] > max_) max_ = x_i[j];
        }
        
        float sum_exp = 0.0f;
        for (int j=0; j<dim; j++){
            float exp_ = expf(x_i[j] - max_);
            x_i[j] = exp_;
            sum_exp += exp_;
        }

        for (int j=0; j<dim; j++){
            x_i[j] /= sum_exp;
        }
    }
}

torch::Tensor softmax_forward(torch::Tensor& input, int axis){

    torch::Tensor output = torch::empty_like(input);
    float* x = input.data_ptr<float>();
    float* y = output.data_ptr<float>();

    int batch_size = 1;
    for (int i=0; i<input.dim()-1; i++){
        batch_size *= input.size(i);
    }
    int dim = input.size(input.dim()-1);
    softmax(x, y, dim, batch_size);
    return output;
}


torch::Tensor softmax_backward(torch::Tensor& output_grad, torch::Tensor& output, int axis){
    // TO DO: ADD COMPABILITY WITH OTHER AXIS
    torch::Tensor input_grad = torch::empty_like(output);
    float* dx = input_grad.data_ptr<float>();
    float* dout = output_grad.data_ptr<float>();
    float* y = output.data_ptr<float>();

    int batch_size = 1;
    for (int i=0; i<output.dim()-1; i++){
        batch_size *= output.size(i);
    }
    int dim = output.size(axis);

    for (int b=0; b < batch_size; b++){
        float* dout_b = dout + b * dim;
        float* y_b = y + b * dim;

        // Reduce operation: sum(dout * y, axis=axis)
        float sum_b = 0.0f;
        for (int d=0; d<dim; d++){
            sum_b += dout_b[d] * y_b[d];
        }
        
        // dout * y - y * sum(dout * y, axis=axis, keepdim=True)
        for (int d=0; d<dim; d++){
            dx[b * dim + d] = dout_b[d] * y_b[d] - y_b[d] * sum_b;
        }
    }

    return input_grad;
}

void softmax_tensor_inplace(torch::Tensor& input, int axis){

    float* x = input.data_ptr<float>();
    int batch_size = 1;
    for (int i=0; i<input.dim()-1; i++){
        batch_size *= input.size(i);
    }
    int dim = input.size(input.dim()-1);
    softmax_inplace(x, dim, batch_size);
}

torch::Tensor scaled_dot_product_attention(torch::Tensor& query, torch::Tensor& key, torch::Tensor& value){
    // ********************************** SETUP *************************************
    int T = query.size(1), L = key.size(1); /*T,L = Tokens in query and key*/
    int B = query.size(0);
    int C = query.size(2);
    float scale = 1.0 / sqrtf(C);
    torch::Tensor output = torch::empty_like(query);
    // Pointers
    float* Q = query.data_ptr<float>(); /*(B,T,C)*/
    float* K = key.data_ptr<float>(); /*(B,L,C)*/
    float* V = value.data_ptr<float>(); /*(B,L,C)*/
    float* attention_weights = new float[B*T*L]; /*(B,T,L)*/
    float* O = output.data_ptr<float>(); /*(B,T,C)*/

    // Attention_weights = (Q @ K.T)/sqrt(C) --- (B,T,C) @ (B,C,L) -> (B,T,L)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* Q_bt = Q + b * T * C + t * C;
            float* attn_bt = attention_weights + b * T * L + t * L;
            
            // Compute QK^T / scale
            float max_val = -INFINITY;
            for (int l = 0; l < L; l++) {
                float dot_product_accum = 0.0f;
                float* K_bl = K + b * L * C + l * C;
                for (int c = 0; c < C; c++) {
                    dot_product_accum += Q_bt[c] * K_bl[c];
                }
                attn_bt[l] = dot_product_accum * scale; 
                max_val = std::max(max_val, attn_bt[l]);
            }

            // Compute exp(QK^T / scale - max_val) for softmax stability
            float sum_exp = 0.0f;
            for (int l = 0; l < L; l++) {
                attn_bt[l] = expf(attn_bt[l] - max_val);
                sum_exp += attn_bt[l];
            }

            // Normalize by sum
            for (int l = 0; l < L; l++) {
                attn_bt[l] /= sum_exp;
            }
        }
    }
    // O = Attention_weights @ V --- (B,T,L) @ (B,L,C) -> (B,T,C)
    bmm(attention_weights, V, O, B, T, L , C);

    delete[] attention_weights;
    return output;
}

void multihead_attention(float* Q, float* K, float* V, float* output,
                         int B, int num_heads, int T, int L, int head_dim
                        ) {
    
    // Inputs: Q (B, T, C), K (B, L, C), V (B, L, C)
    // where C = num_heads * head_dim
    float* attn_weights_row = new float[L]; /*Compute attention weights row one by one, and use them on the fly, no need to store the full matrix*/
    float scale = 1.0 / sqrtf(head_dim);
    int C = head_dim * num_heads;

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int t = 0; t < T; t++) {
                // Q_transposed = Q.view(B, T, num_heads, head_dim).transpose(1,2) ---> (B, num_heads, T, head_dim)
                // Q_transposed[b, h, t] = Q[b, t, h*head_dim]
                float* Q_bht = Q + b * T * C + t * C + h * head_dim;
                
                // Computing a single row of: Q @ K.T / sqrt(head_dim)  ---> (B, num_heads, T, L)[b, h, t, :]
                float max_val = -INFINITY;
                for (int l = 0; l < L; l++) {
                    // K_transposed = K.view(B, L, num_heads, head_dim).transpose(1,2) ---> (B, num_heads, L, head_dim)
                    // K_transposed[b, h, l] = Q[b, l, h*head_dim]
                    float* K_bhl = K + b * L * C + l * C + h * head_dim;
                    
                    float dot_product = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        dot_product += Q_bht[d] * K_bhl[d];
                    }
                    dot_product *= scale;
                    attn_weights_row[l] = dot_product;
                    max_val = std::max(max_val, dot_product); /*Track the max for softmax stability*/
                }
                
                // Softmax over that row
                float sum_exp = 0.0f;
                for (int l=0; l < L; l++){
                    attn_weights_row[l] = expf(attn_weights_row[l] - max_val);
                    sum_exp += attn_weights_row[l];
                }
                for (int l=0; l < L; l++){
                    attn_weights_row[l] /= sum_exp;
                }

                // output_transposed = output.view(B, T, num_heads, head_dim).transpose(1,2) ---> (B, num_heads, T, head_dim)
                // output_transposed[b, h, t] = output[b, t, h*head_dim]
                float* output_bht = output + b * T * C + t * C + h * head_dim;

                // Matmul with V: dot product of that row with the d-th column of V, i.e. V[b, h, :, d], repeat of over all columns of V
                for (int d=0; d < head_dim; d++){
                    // V_transposed = V.view(B, L, num_heads, head_dim).transpose(1,2) ---> (B, num_heads, L, head_dim)
                    // V_transposed[b, h, :, d] = V[b, :, h*head_dim+d]
                    float* V_bh_d = V + b * L * C + h * head_dim + d;

                    // output[b, t, h*head_dim + d] = output_transposed[b, h, t, d] = \sum_d attn_weights[b, h, t, l] * V[b, h, l, d]
                    float dot_product = 0.0f;
                    for (int l=0; l < L; l++){
                        dot_product += attn_weights_row[l] * V_bh_d[l * C];
                    }
                    output_bht[d] = dot_product;
                }
            }
        }
    }

    delete[] attn_weights_row;
}


void multihead_attention_backward(float* Q, float* K, float* V, float* output,
                                int B, int num_heads, int T, int L, int head_dim
                        ) {
    
    // Inputs: Q (B, T, C), K (B, L, C), V (B, L, C)
    // where C = num_heads * head_dim
    float* attn_weights_row = new float[L]; /*Compute attention weights row one by one, and use them on the fly, no need to store the full matrix*/
    float scale = 1.0 / sqrtf(head_dim);
    int C = head_dim * num_heads;

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int t = 0; t < T; t++) {
                // Q_transposed = Q.view(B, T, num_heads, head_dim).transpose(1,2) ---> (B, num_heads, T, head_dim)
                // Q_transposed[b, h, t] = Q[b, t, h*head_dim]
                float* Q_bht = Q + b * T * C + t * C + h * head_dim;
                
                // Computing a single row of: Q @ K.T / sqrt(head_dim)  ---> (B, num_heads, T, L)[b, h, t, :]
                float max_val = -INFINITY;
                for (int l = 0; l < L; l++) {
                    // K_transposed = K.view(B, L, num_heads, head_dim).transpose(1,2) ---> (B, num_heads, L, head_dim)
                    // K_transposed[b, h, l] = Q[b, l, h*head_dim]
                    float* K_bhl = K + b * L * C + l * C + h * head_dim;
                    
                    float dot_product = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        dot_product += Q_bht[d] * K_bhl[d];
                    }
                    dot_product *= scale;
                    attn_weights_row[l] = dot_product;
                    max_val = std::max(max_val, dot_product); /*Track the max for softmax stability*/
                }
                
                // Softmax over that row
                float sum_exp = 0.0f;
                for (int l=0; l < L; l++){
                    attn_weights_row[l] = expf(attn_weights_row[l] - max_val);
                    sum_exp += attn_weights_row[l];
                }
                for (int l=0; l < L; l++){
                    attn_weights_row[l] /= sum_exp;
                }

                // output_transposed = output.view(B, T, num_heads, head_dim).transpose(1,2) ---> (B, num_heads, T, head_dim)
                // output_transposed[b, h, t] = output[b, t, h*head_dim]
                float* output_bht = output + b * T * C + t * C + h * head_dim;

                // Matmul with V: dot product of that row with the d-th column of V, i.e. V[b, h, :, d], repeat of over all columns of V
                for (int d=0; d < head_dim; d++){
                    // V_transposed = V.view(B, L, num_heads, head_dim).transpose(1,2) ---> (B, num_heads, L, head_dim)
                    // V_transposed[b, h, :, d] = V[b, :, h*head_dim+d]
                    float* V_bh_d = V + b * L * C + h * head_dim + d;

                    // output[b, t, h*head_dim + d] = output_transposed[b, h, t, d] = \sum_d attn_weights[b, h, t, l] * V[b, h, l, d]
                    float dot_product = 0.0f;
                    for (int l=0; l < L; l++){
                        dot_product += attn_weights_row[l] * V_bh_d[l * C];
                    }
                    output_bht[d] = dot_product;
                }
            }
        }
    }

    delete[] attn_weights_row;
}

torch::Tensor mha(torch::Tensor& query, torch::Tensor& key, torch::Tensor& value, int num_heads){
    // Multi-head attention
    int B = query.size(0);
    int T = query.size(1), L = key.size(1);
    int C = query.size(2);
    assert(C % num_heads == 0);
    int head_dim = C / num_heads;

    torch::Tensor output = torch::empty_like(query);
    // Pointers
    float* Q = query.data_ptr<float>();
    float* K = key.data_ptr<float>();
    float* V = value.data_ptr<float>();
    float* O = output.data_ptr<float>();

    multihead_attention(Q, K, V, O, B, num_heads, T, L, head_dim);
    return output;
}

torch::Tensor sinusoidal_embedding(torch::Tensor& timesteps, int emb_dim){

    // ********************************** SETUP *************************************
    assert(timesteps.dim() == 1);
    int B = timesteps.size(0);
    torch::Tensor output = torch::empty({B, emb_dim});
    // Pointers
    float* y = output.data_ptr<float>();
    float* t = timesteps.data_ptr<float>();


    // ************************** MAIN PART OF THE CODE *****************************
    float log_10000 = logf(10000.0f);
    // emb(t, 2i) = cos(t/(10000**(2i/d))),  emb(t, 2i+1) = cos(t/(10000**(2i/d)))
    for (int b=0; b<B; b++){
        float time = t[b];
        float* y_b = y + b * emb_dim;
        for (int i=0; i<emb_dim/2; i++){
            float freq = expf(-log_10000 * 2 * i / emb_dim);
            y_b[i] = cosf(freq * time);
            y_b[emb_dim/2+i] = sinf(freq * time);
        }
    }
    return output;
}

torch::Tensor positional_embedding2D(int emb_dim, int height, int width, int base_size){

    // ********************************** SETUP *************************************
    assert(emb_dim % 4 == 0);
    torch::Tensor positional_embedding = torch::empty({height * width, emb_dim}, torch::dtype(torch::kFloat64)); /*TO DO: CHANGE TO FLOAT64*/
    double* emb = positional_embedding.data_ptr<double>();


    // ************************** MAIN PART OF THE CODE *****************************
    double log_10000 = logf(10000.0f);
    for (int w=0; w<width; w++){
        double pos_w = (double) w / width; /*double in [0,1], encoding the width position of a pixel*/
        for (int h=0; h<height; h++){
            double pos_h = (double) h / height; /*double in [0,1], encoding the height position of a pixel*/
            double* emb_wh1 = emb + w * height * emb_dim + h * emb_dim;
            double* emb_wh2 = emb + w * height * emb_dim + h * emb_dim + (1*emb_dim/4);
            double* emb_wh3 = emb + w * height * emb_dim + h * emb_dim + (2*emb_dim/4);
            double* emb_wh4 = emb + w * height * emb_dim + h * emb_dim + (3*emb_dim/4);
            for (int i=0; i<emb_dim/4; i++){
                double freq = expf(-log_10000 * 4 * i / emb_dim);
                emb_wh1[i] = sinf(pos_h * freq);
                emb_wh2[i] = cosf(pos_h * freq);
                emb_wh3[i] = sinf(pos_w * freq);
                emb_wh4[i] = cosf(pos_w * freq);
            }
        }
    }
    return positional_embedding;
}

std::pair<int, int> get_output_size(int in_height, int in_width, int kernel_height, int kernel_width, int stride, int padding){
    int out_height = (in_height + 2*padding - kernel_height) / stride + 1;
    int out_width = (in_width + 2*padding - kernel_width) / stride + 1;
    return {out_height, out_width};
}

void convolve2d_single_channel(
    float* input, float* kernel, float* output, int stride, int padding, 
    int kernel_height, int kernel_width, int in_height, int in_width, int out_height, int out_width){
                
    // Convolves input with kernel, the result is written in output (initialized with 0s)
    // batch_size = in_channels = out_channels = 1, the input is expected to be of shape (1,1, in_height, in_width)

    for (int h = 0; h < out_height; h++){
        for (int w = 0 ; w < out_width; w++){
            float accum = 0.0f;
            // inner 2D dot product
            for (int row=0; row < kernel_height; row++){
                for (int col=0; col < kernel_width; col++){
                    int h_padded = h * stride + row - padding;
                    int w_padded = w * stride + col - padding;
                    if (h_padded >= 0 && h_padded < in_height && w_padded >= 0 && w_padded < in_width){
                        accum += kernel[row * kernel_width + col] * input[h_padded * in_width + w_padded];
                    }
                }
            }
            /*output[0,0,h,w]*/
            output[h * out_width + w] += accum;
        }
    }
}


torch::Tensor conv2d_forward(torch::Tensor& input, torch::Tensor& kernel, torch::Tensor& bias, int stride, int padding){
    // ********************************** SETUP *************************************
    assert(kernel.dim() == 4);
    assert(input.dim()==4);
    int out_channels = kernel.size(0);
    int in_channels = kernel.size(1);
    int kernel_height = kernel.size(2);
    int kernel_width = kernel.size(3);
    int batch_size = input.size(0);
    int in_height = input.size(2);
    int in_width = input.size(3);

    // Formula from: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    // Simplified because we always use dilation=1
    int out_height = (in_height + 2 * padding - kernel_height) / stride + 1; /*Integer divison floors the result*/
    int out_width = (in_width + 2 * padding - kernel_width) / stride + 1; 
    torch::Tensor output = torch::zeros({batch_size, out_channels, out_height, out_width});
    // data pointers
    float* x = input.data_ptr<float>();
    float* y = output.data_ptr<float>();
    float* kernel_ = kernel.data_ptr<float>();
    float* bias_ = bias.data_ptr<float>();


    // ************************** MAIN PART OF THE CODE *****************************
    // out[b, c_out] = bias[c_out] + sum_{c_in} kernel[out_c, c_in] ∗ input[b, c_in]
    for (int b = 0; b < batch_size; b++){
        for (int out_c = 0; out_c < out_channels; out_c++){
            // Pointer at output at output[b, out_c]
            float* y_bc = y + b * out_channels * out_height * out_width + out_c * out_height * out_width;
            for (int in_c = 0; in_c < in_channels; in_c++){
                // Pointer to input at input[b, in_c]
                float* x_bc = x + b * in_channels * in_height * in_width + in_c * in_height * in_width;
                // Pointer to kernel at kernel[out_c, in_c]
                float* kernel_cc = kernel_ + out_c * in_channels * kernel_height * kernel_width + in_c * kernel_height * kernel_width;
                
                // Single-channel 2d convolution operation: kernel[out_c, c_in] ∗ input[b, c_in]
                convolve2d_single_channel(x_bc, kernel_cc, y_bc, stride, padding, kernel_height, kernel_width, in_height, in_width, out_height, out_width);
            }

            //  Adding the bias
            for (int h=0; h < out_height; h++){
                for (int w=0; w < out_width; w++){
                    y_bc[h * out_width + w] += bias_[out_c];
                }
            }
        }
    }

    return output;
}

void mlp(float*x, float* output, float* W1, float* b1, float* h1, float* W2, float* b2, int B, int C_in, int C_hid, int C_out, 
    std::function<void (float*, float*, long long)> activation){

    // Linear -> activation -> Linear
    linear(x, W1, b1, h1, B, C_in, C_hid);
    activation(h1, h1, B*C_hid);
    linear(h1, W2, b2, output, B, C_hid, C_out);
}

torch::Tensor MLP(
    torch::Tensor& input, torch::Tensor& weight1, torch::Tensor& bias1, torch::Tensor& weight2, torch::Tensor& bias2,
    std::string activation, int B, int C_in, int C_hid, int C_out
){
    assert(activation == "GELU" || activation == "gelu" || activation == "silu" || activation == "SILU" || activation == "SiLU");
    torch::Tensor output = torch::empty({B, C_out});
    // Pointers
    float* x = input.data_ptr<float>();
    float* W1 = weight1.data_ptr<float>();
    float* b1 = bias1.data_ptr<float>();
    float* h1 = new float[B*C_hid];
    float* W2 = weight2.data_ptr<float>();
    float* b2 = bias2.data_ptr<float>();
    float* y = output.data_ptr<float>();


    // y = act(x @ W1 + b1) @ W2 + b2 --- shape: (B, C_out)
    std::function<void (float*, float*, long long)> act;
    if (activation == "GELU" || activation == "gelu") act = gelu;
    else act = silu;

    mlp(x, y, W1, b1, h1, W2, b2, B, C_in, C_hid, C_out, act);

    delete[] h1;
    return output;
}

float get_max(float* arr, long long size){
    float max_value = -INFINITY;
    for (int i=0; i<size; ++i){
        max_value = (arr[i] > max_value)? arr[i] : max_value;
    }
    return max_value;
}


void scale_and_shift(float* input, float* output, float* shift, float* scale, int B, int T, int C){
    // output = x * (1 + scale[:, None, :]) + shift[:, None, :]
    for (int b=0; b<B; b++){
        for (int c=0; c<C; c++){
            float* x_b_c = input + b * T * C + c; /*input[b,:,c]*/
            float* out_b_c = output + b * T * C + c; /*output[b,:,c]*/
            float scale_bc = scale[b * C + c]; /*scale[b,c]*/
            float shift_bc = shift[b * C + c];

            for (int t=0; t<T; t++){
                out_b_c[t * C] = x_b_c[t * C] * (1.0f + scale_bc) + shift_bc;
            }
        }
    }
}

void gating_mechanism(float* input, float* output, float* gate, int B, int T, int C){
    // output = input * gate[:, None, :]
    for (int b=0; b<B; b++){
        for (int c=0; c<C; c++){
            float* x_b_c = input + b * T * C + c; /*input[b,:,c]*/
            float* out_b_c = output + b * T * C + c; /*output[b,:,c]*/
            float gate_bc = gate[b * C + c]; /*scale[b,c]*/

            for (int t=0; t<T; t++){
                out_b_c[t * C] = x_b_c[t * C] * gate_bc;
            }
        }
    }
}


void adaptative_layernorm_zero(
    float* input, std::vector<float*> outputs, std::vector<float*> weights, std::vector<float*> biases, int B, int emb_dim){
    // SiLU + Linear forward

    long long n_elem = B*emb_dim;
    float* temp = new float[n_elem];
    silu(input, temp, n_elem);

    for (int i=0; i<outputs.size(); i++){
        // if nullptr, don't do operations (useful when switching from using/not using dual attention layers or when discarding context embeddings)
        if (outputs[i] == nullptr || weights[i] == nullptr || biases[i] == nullptr){
            std::cout << "Skipping ada linear_" << i << std::endl;
            continue;
        }
        linear(temp, weights[i], biases[i], outputs[i], B, emb_dim, emb_dim);
    }
    delete[] temp;
    temp = NULL;
}

void concatenate_along_tokens(float* x, float* y, float* output, int B, int Tx, int Ty, int C){
    // Crap version of torch::cat((x, y), dim=1)
    // Concatenate x and y along T axis
    int T = Tx + Ty;
    for (int b=0; b<B; b++){
        // Pointers
        float* out_bt = output + b * T * C;
        float* x_bt = x + b * Tx * C;
        float* y_bt = y + b * Ty * C;

        // Copy x into the first Tx tokens of output
        for (int t=0; t<Tx; t++){
            for (int c=0; c<C; c++){
                out_bt[c] = x_bt[c];
            }
            out_bt += C;
            x_bt += C;
        }
        // copy y in the remaining tokens
        for (int t=0; t<Ty; t++){
            for (int c=0; c<C; c++){
                out_bt[c] = y_bt[c];
            }
            out_bt += C;
            y_bt += C;
        }
    }
}

void split_along_tokens(float* input, float* split1, float* split2, int B, int T1, int T2, int C){
    // Split x (B,T,C) into x[:,:T1,:]  and x[:,T1:,:]
    int T = T1 + T2;
    for (int b=0; b<B; b++){
        // Pointers
        float* x_bt = input + b * T * C;
        float* s1_bt = split1 + b * T1 * C;
        float* s2_bt = split2 + b * T2 * C;

        // copy over the first T1 tokens of input into split1
        for (int t=0; t<T1; t++){
            for (int c=0; c<C; c++){
                s1_bt[c] = x_bt[c];
            }
            x_bt += C;
            s1_bt += C;
        }
        // copy the rest into split2
        for (int t=0; t<T2; t++){
            for (int c=0; c<C; c++){
                s2_bt[c] = x_bt[c];
            }
            x_bt += C;
            s2_bt += C;
        }
    }
}

void copy(float* source, float* dest, long long size){
    for (int i=0; i<size; i++){
        dest[i] = source[i];
    }
}

void print_first5(float* arr){
    for (int i=0; i<5; i++){
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

void split_dim0(float* input, std::vector<float*> splits, int n_splits, int dim0){
    // Splits the 1D-array input into 'n_splits' splits in 'splits'

    const int dim_per_split = dim0 / n_splits;
    float* input_s = input; /*Pointer to input[s], where s is the split index*/

    for (int s=0; s<n_splits; s++){
        for (int d=0; d<dim_per_split; d++){
            splits[s][d] = input_s[d];
        }
        input_s += dim_per_split;
    }
}

void split_dim1(float* input, std::vector<float*> splits, int n_splits, int dim0, int dim1){
    if (dim1 % n_splits != 0) {
        throw std::invalid_argument("dim1 must be divisible by n_splits");
    }
    const int dim1_per_split = dim1 / n_splits;

    for (int b=0; b<dim0; b++){
        for (int s=0; s<n_splits; s++){
            float* split_sb = splits[s] + b * dim1_per_split;
            float* input_sb = input + b * dim1 + s * dim1_per_split;
            for (int d=0; d<dim1_per_split; d++){
                split_sb[d] = input_sb[d];
            }
        }
    }
}

void dit_block(
    // Inputs, outputs
    float* y, float* c, float* x, float* c_out, float* x_out, float* x_out_dual,
    // Context tensor parameters
    // adaptative layernorm linear weights & biases
    std::vector<float*>& c_ada_lnorm_weights,
    std::vector<float*>& c_ada_lnorm_biases,
    float* c_Wq, float* c_Wk, float* c_Wv, float* c_Wo, float* c_bias_q, float* c_bias_k, float* c_bias_v, float* c_bias_o, /*attention*/
    float* c_rms_Wq, float* c_rms_Wk, /*rms norm weight*/
    float* c_mlp_W1, float* c_mlp_b1, float* c_mlp_W2, float* c_mlp_b2, /*mlp*/
    // // Latent tensor parameters
    std::vector<float*>& x_ada_lnorm_weights,
    std::vector<float*>& x_ada_lnorm_biases,
    float* x_Wq, float* x_Wk, float* x_Wv, float* x_Wo, float* x_bias_q, float* x_bias_k, float* x_bias_v, float* x_bias_o,
    float* x_Wq_dual, float* x_Wk_dual, float* x_Wv_dual, float* x_Wo_dual, float* x_bias_q_dual, float* x_bias_k_dual, float* x_bias_v_dual, float* x_bias_o_dual,
    float* x_rms_Wq, float* x_rms_Wk, float* x_rms_Wq_dual, float* x_rms_Wk_dual,
    float* x_mlp_W1, float* x_mlp_b1, float* x_mlp_W2, float* x_mlp_b2, 
    // Intermediate tensors
    float* c_query, float* c_key, float* c_value, float* x_query, float* x_key, float* x_value, float* query, float* key, float* value, 
    float* c_mlp_hidden, float* x_mlp_hidden,
    std::vector<float*>& c_shift_scale_gate,
    std::vector<float*>& x_shift_scale_gate,
    // Others
    int B, int Tc, int Tx, int emb_dim, int attn_heads, int mlp_expand, bool use_dual_attention, bool use_kqnorm, bool discard_context){
    
    // ***************************************************** SETUP ***********************************************************
    // Tx = number of tokens in latent vector (the image)
    // Tc = number of tokens in context embeddings (the text)
    // In the following, c (lower-case) will always refer to the context embeddings
    // and x refers to the latent vector
    const int T = Tx + Tc;
    const int head_dim = emb_dim / attn_heads;
    // ***************************************************** MAIN PART ***********************************************************
    // *********************************** Pre-attention Context **********************************
    adaptative_layernorm_zero(y, c_shift_scale_gate, c_ada_lnorm_weights, c_ada_lnorm_biases, B, emb_dim);
    // c_out = layernorm(c, weight=None, bias=None, save_for_backward=False)
    layernorm(c, c_out, nullptr, nullptr, nullptr, nullptr, 1e-6, B, Tc, emb_dim);
    // c_out = c_out * (1.0 + scale_attn_context) + shift_attn_context
    if (!discard_context) scale_and_shift(c_out, c_out, c_shift_scale_gate[0], c_shift_scale_gate[1], B, Tc, emb_dim);
    else scale_and_shift(c_out, c_out, c_shift_scale_gate[1], c_shift_scale_gate[0], B, Tc, emb_dim);
    // Wqkv projections
    linear(c_out, c_Wq, c_bias_q, c_query, B*Tc, emb_dim, emb_dim);
    linear(c_out, c_Wk, c_bias_k, c_key, B*Tc, emb_dim, emb_dim);
    linear(c_out, c_Wv, c_bias_v, c_value, B*Tc, emb_dim, emb_dim);
    // RMSNorm q, k
    if (use_kqnorm){
        rmsnorm_group(c_query, c_query, c_rms_Wq, 1e-6, attn_heads, B, Tc, emb_dim);
        rmsnorm_group(c_key, c_key, c_rms_Wk, 1e-6, attn_heads, B, Tc, emb_dim);
    }

    // *********************************** Pre-attention Latent **********************************
    adaptative_layernorm_zero(y, x_shift_scale_gate, x_ada_lnorm_weights, x_ada_lnorm_biases, B, emb_dim);
    layernorm(x, x_out, nullptr, nullptr, nullptr, nullptr, 1e-6, B, Tx, emb_dim);
    // x_out = x_out * (1.0 + scale_attn_latent) + shift_attn_latent
    if (use_dual_attention) scale_and_shift(x_out, x_out_dual, x_shift_scale_gate[6], x_shift_scale_gate[7], B, Tx, emb_dim);
    scale_and_shift(x_out, x_out, x_shift_scale_gate[0], x_shift_scale_gate[1], B, Tx, emb_dim);
    // Wqkv projections
    linear(x_out, x_Wq, x_bias_q, x_query, B*Tx, emb_dim, emb_dim);
    linear(x_out, x_Wk, x_bias_k, x_key, B*Tx, emb_dim, emb_dim);
    linear(x_out, x_Wv, x_bias_v, x_value, B*Tx, emb_dim, emb_dim);
    // to_qkv(x_out, x_Wqkv, x_bias_qkv, x_query, x_key, x_value, B*Tx, emb_dim, emb_dim);
    // RMSNorm q, k
    if (use_kqnorm){
        rmsnorm_group(x_query, x_query, x_rms_Wq, 1e-6, attn_heads, B, Tx, emb_dim);
        rmsnorm_group(x_key, x_key, x_rms_Wk, 1e-6, attn_heads, B, Tx, emb_dim);
    }
    
    // ************************************* Attention ********************************************
    // concatenate context and latent to allow text-image tokens attend to each other
    concatenate_along_tokens(c_query, x_query, query, B, Tc, Tx, emb_dim);
    concatenate_along_tokens(c_key, x_key, key, B, Tc, Tx, emb_dim);
    concatenate_along_tokens(c_value, x_value, value, B, Tc, Tx, emb_dim);
    multihead_attention(query, key, value, query, B, attn_heads, T, T, head_dim); /*Output of self-attention is written back into query*/
    // Split the result back into context and latent
    split_along_tokens(query, c_query, x_query, B, Tc, Tx, emb_dim);
    // W_out projections
    if (!discard_context) linear(c_query, c_Wo, c_bias_o, c_out, B*Tc, emb_dim, emb_dim);
    linear(x_query, x_Wo, x_bias_o, x_out, B*Tx, emb_dim, emb_dim);

    // ************************************* Dual Attention ********************************************

    if (use_dual_attention){
        linear(x_out_dual, x_Wq_dual, x_bias_q_dual, x_query, B*Tx, emb_dim, emb_dim);
        linear(x_out_dual, x_Wk_dual, x_bias_k_dual, x_key, B*Tx, emb_dim, emb_dim);
        linear(x_out_dual, x_Wv_dual, x_bias_v_dual, x_value, B*Tx, emb_dim, emb_dim);
        if (use_kqnorm){
            rmsnorm_group(x_query, x_query, x_rms_Wq_dual, 1e-6, attn_heads, B, Tx, emb_dim);
            rmsnorm_group(x_key, x_key, x_rms_Wk_dual, 1e-6, attn_heads, B, Tx, emb_dim);
        }
        multihead_attention(x_query, x_key, x_value, x_query, B, attn_heads, Tx, Tx, head_dim);
        linear(x_query, x_Wo_dual, x_bias_o_dual, x_out_dual, B*Tx, emb_dim, emb_dim);
        gating_mechanism(x_out_dual, x_out_dual, x_shift_scale_gate[8], B, Tx, emb_dim);
    }

    // *********************************** Context MLP **********************************
    if (!discard_context){
        gating_mechanism(c_out, c_out, c_shift_scale_gate[2], B, Tc, emb_dim);
        // Skip connection: c = c + c_out
        add(c_out, c, c, B*Tc*emb_dim);
        layernorm(c, c_out, nullptr, nullptr, nullptr, nullptr, 1e-6, B, Tc, emb_dim);
        scale_and_shift(c_out, c_out, c_shift_scale_gate[3] , c_shift_scale_gate[4], B, Tc, emb_dim);
        mlp(c_out, c_out, 
            c_mlp_W1, c_mlp_b1, c_mlp_hidden, c_mlp_W2, c_mlp_b2,
            B*Tc, emb_dim, mlp_expand*emb_dim, emb_dim, gelu_tanh
        );
        gating_mechanism(c_out, c_out, c_shift_scale_gate[5], B, Tc, emb_dim);
        add(c_out, c, c_out, B*Tc*emb_dim);
    }


    // *********************************** Latent MLP **********************************
    gating_mechanism(x_out, x_out, x_shift_scale_gate[2], B, Tx, emb_dim);
    // Skip connection: x = x + x_out + x_out_dual
    add(x_out, x, x, B*Tx*emb_dim);
    if (use_dual_attention) add(x_out_dual, x, x, B*Tx*emb_dim);
    layernorm(x, x_out, nullptr, nullptr, nullptr, nullptr, 1e-6, B, Tx, emb_dim);
    scale_and_shift(x_out, x_out, x_shift_scale_gate[3], x_shift_scale_gate[4], B, Tx, emb_dim);
    mlp(x_out, x_out, 
        x_mlp_W1, x_mlp_b1, x_mlp_hidden, x_mlp_W2, x_mlp_b2,
        B*Tx, emb_dim, mlp_expand*emb_dim, emb_dim, gelu_tanh
    );

    gating_mechanism(x_out, x_out, x_shift_scale_gate[5], B, Tx, emb_dim);
    add(x_out, x, x_out, B*Tx*emb_dim);

}



std::vector<torch::Tensor> DiT_block_forward(
    // Inputs
    torch::Tensor& temp_embeddings, torch::Tensor& context_embeddings, torch::Tensor& latent_vector,
    // Context tensor parameters
    torch::Tensor& context_ada_linear_weight, torch::Tensor& context_ada_linear_bias,
    torch::Tensor& context_Wq, torch::Tensor& context_Wk, torch::Tensor& context_Wv, torch::Tensor& context_bq, torch::Tensor& context_bk, torch::Tensor& context_bv,
    torch::Tensor& context_rmsnorm_Wq, torch::Tensor& context_rmsnorm_Wk,
    torch::Tensor& context_Wo, torch::Tensor& context_bo,
    torch::Tensor& context_mlp_weight1, torch::Tensor& context_mlp_bias1, torch::Tensor& context_mlp_weight2, torch::Tensor& context_mlp_bias2,
    // Latent tensor parameters
    torch::Tensor& latent_ada_linear_weight, torch::Tensor& latent_ada_linear_bias,
    torch::Tensor& latent_Wq, torch::Tensor& latent_Wk, torch::Tensor& latent_Wv, torch::Tensor& latent_bq, torch::Tensor& latent_bk, torch::Tensor& latent_bv,
    torch::Tensor& latent_Wq_dual, torch::Tensor& latent_Wk_dual, torch::Tensor& latent_Wv_dual, torch::Tensor& latent_bq_dual, torch::Tensor& latent_bk_dual, torch::Tensor& latent_bv_dual,
    torch::Tensor& latent_rmsnorm_Wq, torch::Tensor& latent_rmsnorm_Wk, torch::Tensor& latent_rmsnorm_Wq_dual, torch::Tensor& latent_rmsnorm_Wk_dual,
    torch::Tensor& latent_Wo, torch::Tensor& latent_bo, torch::Tensor& latent_Wo_dual, torch::Tensor& latent_bo_dual,
    torch::Tensor& latent_mlp_weight1, torch::Tensor& latent_mlp_bias1, torch::Tensor& latent_mlp_weight2, torch::Tensor& latent_mlp_bias2,
    // Others
    int B, int Tc, int Tx, int emb_dim, int attn_heads, int mlp_expand, bool use_dual_attention, bool use_kqnorm, bool discard_context){
    
    // ***************************************************** SETUP ***********************************************************
    // In the following, c (lower-case) will always refer to the context embeddings
    // and x refers to the latent vector
    int T = Tc + Tx;
    torch::Tensor context_embeddings_out = torch::empty_like(context_embeddings);
    torch::Tensor latent_vector_out = torch::empty_like(latent_vector);
    // Pointers
    float* y = temp_embeddings.data_ptr<float>();
    float* c = context_embeddings.data_ptr<float>();
    float* x = latent_vector.data_ptr<float>();
    float* c_out = context_embeddings_out.data_ptr<float>();
    float* x_out = latent_vector_out.data_ptr<float>();

    // Intermediate activation pointers
    float* x_out_dual = use_dual_attention ? new float[B*Tx*emb_dim] : nullptr;
    // Adaptative layer norm
    std::vector<float*> c_shift_scale_gate;
    std::vector<float*> x_shift_scale_gate;
    std::vector<float*> c_ada_weights;
    std::vector<float*> c_ada_biases;
    std::vector<float*> x_ada_weights;
    std::vector<float*> x_ada_biases;
    for (int i=0; i<6; i++){
        if (discard_context && i>=2){
            c_shift_scale_gate.push_back(nullptr);
            c_ada_weights.push_back(nullptr);
            c_ada_biases.push_back(nullptr);
        } else{
            c_shift_scale_gate.push_back(new float[B*emb_dim]);
            c_ada_weights.push_back(new float[emb_dim * emb_dim]);
            c_ada_biases.push_back(new float[emb_dim]);
        }
        x_shift_scale_gate.push_back(new float[B*emb_dim]);
        x_ada_weights.push_back(new float[emb_dim * emb_dim]);
        x_ada_biases.push_back(new float[emb_dim]);
    }
    for (int i=0; i<3; i++){
        if (use_dual_attention){
            x_shift_scale_gate.push_back(new float[B*emb_dim]);
            x_ada_weights.push_back(new float[emb_dim * emb_dim]);
            x_ada_biases.push_back(new float[emb_dim]);
        }
        else {
            x_shift_scale_gate.push_back(nullptr);
            x_ada_weights.push_back(nullptr);
            x_ada_biases.push_back(nullptr);
        }
    }

    int context_splits = (discard_context) ? 2 : 6;
    split_dim0(context_ada_linear_bias.data_ptr<float>(), c_ada_biases, context_splits, context_splits*emb_dim);
    split_dim1(context_ada_linear_weight.data_ptr<float>(), c_ada_weights, context_splits, emb_dim, context_splits*emb_dim);
    int latent_splits = (use_dual_attention) ? 9 : 6;
    split_dim0(latent_ada_linear_bias.data_ptr<float>(), x_ada_biases, latent_splits, latent_splits*emb_dim);
    split_dim1(latent_ada_linear_weight.data_ptr<float>(), x_ada_weights, latent_splits, emb_dim, latent_splits*emb_dim);

    // attention
    float* qc = new float[B*Tc*emb_dim];
    float* kc = new float[B*Tc*emb_dim];
    float* vc = new float[B*Tc*emb_dim];
    float* qx = new float[B*Tx*emb_dim];
    float* kx = new float[B*Tx*emb_dim];
    float* vx = new float[B*Tx*emb_dim];
    float* Q = new float[B*T*emb_dim];
    float* K = new float[B*T*emb_dim];
    float* V = new float[B*T*emb_dim];
    // mlp
    float* mlp_hidden_c = new float[B*Tc*mlp_expand*emb_dim];
    float* mlp_hidden_x = new float[B*Tx*mlp_expand*emb_dim];


    dit_block(
        y, c, x, c_out, x_out, x_out_dual,
        // Context
        c_ada_weights,
        c_ada_biases,
        context_Wq.data_ptr<float>(), context_Wk.data_ptr<float>(), context_Wv.data_ptr<float>(), context_Wo.data_ptr<float>(), 
        context_bq.data_ptr<float>(), context_bk.data_ptr<float>(), context_bv.data_ptr<float>(), context_bo.data_ptr<float>(),
        context_rmsnorm_Wq.data_ptr<float>(), context_rmsnorm_Wk.data_ptr<float>(), 
        context_mlp_weight1.data_ptr<float>(), context_mlp_bias1.data_ptr<float>(), context_mlp_weight2.data_ptr<float>(), context_mlp_bias2.data_ptr<float>(),
        // Latent
        x_ada_weights,
        x_ada_biases,
        latent_Wq.data_ptr<float>(), latent_Wk.data_ptr<float>(), latent_Wv.data_ptr<float>(), latent_Wo.data_ptr<float>(), 
        latent_bq.data_ptr<float>(), latent_bk.data_ptr<float>(), latent_bv.data_ptr<float>(), latent_bo.data_ptr<float>(),
        latent_Wq_dual.data_ptr<float>(), latent_Wk_dual.data_ptr<float>(), latent_Wv_dual.data_ptr<float>(), latent_Wo_dual.data_ptr<float>(),
        latent_bq_dual.data_ptr<float>(), latent_bk_dual.data_ptr<float>(), latent_bv_dual.data_ptr<float>(), latent_bo_dual.data_ptr<float>(),
        latent_rmsnorm_Wq.data_ptr<float>(), latent_rmsnorm_Wk.data_ptr<float>(), latent_rmsnorm_Wq_dual.data_ptr<float>(), latent_rmsnorm_Wk_dual.data_ptr<float>(), 
        latent_mlp_weight1.data_ptr<float>(), latent_mlp_bias1.data_ptr<float>(), latent_mlp_weight2.data_ptr<float>(), latent_mlp_bias2.data_ptr<float>(),
        // // Intermediate tensors
        qc, kc, vc, qx, kx, vx, Q, K, V, mlp_hidden_c, mlp_hidden_x,
        c_shift_scale_gate,
        x_shift_scale_gate,
        // Others
        B, Tc, Tx, emb_dim, attn_heads, mlp_expand, use_dual_attention, use_kqnorm, discard_context
    );

    delete[] x_out_dual;
    delete[] qc;
    delete[] kc;
    delete[] vc;
    delete[] qx;
    delete[] kx;
    delete[] vx;
    delete[] Q;
    delete[] K;
    delete[] V;
    delete[] mlp_hidden_c;
    delete[] mlp_hidden_x;
    for (auto& arr: c_ada_weights){
        delete[] arr;
    }
    for (auto& arr: x_ada_weights){
        delete[] arr;
    }
    for (auto& arr: c_ada_biases){
        delete[] arr;
    }
    for (auto& arr: x_ada_biases){
        delete[] arr;
    }
    for (auto& arr: c_shift_scale_gate){
        delete[] arr;
    }
    for (auto& arr: x_shift_scale_gate){
        delete[] arr;
    }
    // return c_out, x_out as tensors
    return {context_embeddings_out, latent_vector_out};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("layer_norm", &LayerNorm_forward, "LayerNorm forward");
    m.def("layer_norm_backward", &LayerNorm_backward, "LayerNorm backward");
    m.def("rms_norm", &RMSNorm_forward, "RMSNorm forward");
    m.def("rms_norm_backward", &RMSNorm_backward, "RMSNorm backward");
    m.def("gelu", &GELU_forward, "GELU forward");
    m.def("gelu_backward", &GELU_backward, "GELU backward");
    m.def("silu", &SiLU_forward, "SiLU forward");
    m.def("silu_backward", &SiLU_backward, "SiLU backward");
    m.def("softmax", &softmax_forward, "softmax");
    m.def("softmax_", &softmax_tensor_inplace, "In-place softmax");
    m.def("softmax_backward", &softmax_backward, "Softmax backward");
    m.def("linear", &Linear_forward, "Linear forward");
    m.def("linear_backward", &Linear_backward, "Linear backward");
    m.def("conv2d", &conv2d_forward, "Conv2d forward");
    m.def("sinusoidal_embedding", &sinusoidal_embedding, "Temporal sinusoidal embeddings");
    m.def("positional_embedding", &positional_embedding2D, "2D positional embeddings");
    m.def("scaled_dot_product_attention", &scaled_dot_product_attention, "single-head self attention");
    m.def("mha", &mha, "multi-head self attention");
    m.def("mlp", &MLP, "MLP");
    m.def("Dit_block", &DiT_block_forward, "Stable diffusion's DiT block");
}