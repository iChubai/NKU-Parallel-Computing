#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <sstream>
#include <algorithm>

// 编译文件
// hipcc sourcefile_mlp.cpp -o mlp_full_dcu
// 执行文件
// ./mlp_full_dcu 或者 hipprof ./mlp_full_dcu

// 预定义参数，可根据需求修改
#define INPUT_DIM 10
#define HIDDEN_DIM 32
#define OUTPUT_DIM 1
#define BATCH_SIZE 256
#define EPOCHS 200
#define LEARNING_RATE 1e-4

#define HIP_CHECK(error) \
    do { \
        hipError_t err = error; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 以下函数和main函数均不为固定形式，可自行按照需求修改

// HIP kernels函数形式，需要自行设计
__global__ void matmul(const double* A, const double* B, double* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0.0;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void relu_forward(double* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = fmax(0.0, x[idx]);
    }
}

__global__ void compute_output_grad(const double* pred, const double* target, double* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = pred[idx] - target[idx];
    }
}

__global__ void compute_relu_backward(double* delta, const double* activ, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (activ[idx] <= 0) {
            delta[idx] = 0;
        }
    }
}

__global__ void matmul_backward_weights(const double* upstream_grad, const double* layer_input, double* weight_grad, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < K) {
        double sum = 0.0;
        for (int i = 0; i < M; ++i) {
            sum += layer_input[i * N + row] * upstream_grad[i * K + col];
        }
        weight_grad[row * K + col] = sum;
    }
}

__global__ void matmul_backward_input(const double* upstream_grad, const double* weights, double* input_grad, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0.0;
        for (int i = 0; i < K; ++i) {
            sum += upstream_grad[row * K + i] * weights[col * K + i];
        }
        input_grad[row * N + col] = sum;
    }
}

__global__ void compute_mse_loss(const double* pred, const double* target, double* loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double diff = pred[idx] - target[idx];
        loss[idx] = diff * diff * 0.5;
    }
}

__global__ void sgd_update(double* weights, const double* grad, double lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= lr * grad[idx];
    }
}

__global__ void add_bias(double* output, const double* bias, int M, int N) {
    // output is M x N, bias is 1 x N (or just N)
    // Adds bias vector to each row of the output matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        output[row * N + col] += bias[col];
    }
}

__global__ void sum_gradients_for_bias(const double* delta, double* bias_grad, int batch_size, int num_neurons) {
    // delta is (batch_size x num_neurons)
    // bias_grad is (num_neurons)
    // This is a simplified version; a proper reduction is better.
    // This kernel assumes it's launched with enough threads for num_neurons.
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx < num_neurons) {
        double sum = 0.0;
        for (int i = 0; i < batch_size; ++i) {
            sum += delta[i * num_neurons + neuron_idx];
        }
        // Atomic add or careful single write if bias_grad is global sum
        // For now, let's assume bias_grad is an intermediate buffer and averaging happens later or SGD handles it per-element.
        // Or, if this is directly the gradient to be used by SGD:
        bias_grad[neuron_idx] = sum / batch_size; // Averaged gradient
    }
}

// 加载带宽数据
std::vector<double> load_json_bandwidth(const std::string& filename) {
    std::vector<double> bandwidth_data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return bandwidth_data;
    }

    std::string content;
    std::string line;
    while (std::getline(file, line)) {
        content += line;
    }
    file.close();

    content.erase(std::remove_if(content.begin(), content.end(), ::isspace), content.end());
    
    size_t start_pos = content.find('[');
    size_t end_pos = content.rfind(']');

    if (start_pos == std::string::npos || end_pos == std::string::npos || start_pos >= end_pos) {
        std::cerr << "Error: Invalid JSON array format in " << filename << std::endl;
        return bandwidth_data;
    }

    std::string array_content = content.substr(start_pos + 1, end_pos - start_pos - 1);
    std::stringstream ss(array_content);
    std::string item;

    while (std::getline(ss, item, ',')) {
        try {
            bandwidth_data.push_back(std::stod(item));
        } catch (const std::invalid_argument& ia) {
            std::cerr << "Invalid argument: " << ia.what() << " for item: " << item << std::endl;
        } catch (const std::out_of_range& oor) {
            std::cerr << "Out of Range error: " << oor.what() << " for item: " << item << std::endl;
        }
    }
    return bandwidth_data;
}

// 创建数据集
void create_dataset(const std::vector<double>& data,
                    std::vector<double>& X,
                    std::vector<double>& y,
                    int window_size) {
    X.clear();
    y.clear();
    if (data.size() <= window_size) {
        std::cerr << "Error: Data size is too small for the given window size." << std::endl;
        return;
    }

    for (size_t i = 0; i < data.size() - window_size; ++i) {
        for (int j = 0; j < window_size; ++j) {
            X.push_back(data[i + j]);
        }
        y.push_back(data[i + window_size]);
    }
    return;
}

// 数据归一化处理
void normalize_data(std::vector<double>& data, double& min_val, double& max_val) {
    min_val = *std::min_element(data.begin(), data.end());
    max_val = *std::max_element(data.begin(), data.end());
    for (auto& val : data) {
        val = (val - min_val) / (max_val - min_val);
    }
    return;
}

// 数据反归一化处理
void denormalize_data(std::vector<double>& data, double min_val, double max_val) {
    for (auto& val : data) {
        val = val * (max_val - min_val) + min_val;
    }
    return;
}

// ----------------------------- Main -------------------------------
int main() {
        // 读取带宽json文件，并生成测试集和训练集
    std::cout << "[INFO] Loading bandwidth data..." << std::endl;
    std::vector<double> bandwidth_series = load_json_bandwidth("starlink_bw.json");
    if (bandwidth_series.empty()) {
        std::cerr << "[ERROR] Failed to load bandwidth data. Exiting." << std::endl;
        return 1;
    }
    std::cout << "[INFO] Loaded " << bandwidth_series.size() << " bandwidth records." << std::endl;

    double min_val, max_val;
    std::vector<double> normalized_series = bandwidth_series;
    normalize_data(normalized_series, min_val, max_val);
    std::cout << "[INFO] Data normalized. Min: " << min_val << ", Max: " << max_val << std::endl;

    std::vector<double> h_all_X, h_all_y;
    create_dataset(normalized_series, h_all_X, h_all_y, INPUT_DIM);
    if (h_all_X.empty() || h_all_y.empty()) {
        std::cerr << "[ERROR] Failed to create dataset. Exiting." << std::endl;
        return 1;
    }
    size_t total_samples = h_all_y.size();
    std::cout << "[INFO] Created dataset with " << total_samples << " samples." << std::endl;

    // 数据集划分
    const double TRAIN_SPLIT_RATIO = 0.8;
    size_t train_samples = static_cast<size_t>(total_samples * TRAIN_SPLIT_RATIO);
    size_t test_samples = total_samples - train_samples;

    std::vector<double> h_train_X(h_all_X.begin(), h_all_X.begin() + train_samples * INPUT_DIM);
    std::vector<double> h_train_y(h_all_y.begin(), h_all_y.begin() + train_samples);
    std::vector<double> h_test_X(h_all_X.begin() + train_samples * INPUT_DIM, h_all_X.end());
    std::vector<double> h_test_y(h_all_y.begin() + train_samples, h_all_y.end());

    std::cout << "[INFO] Training samples: " << train_samples << std::endl;
    std::cout << "[INFO] Testing samples: " << test_samples << std::endl;

    // MLP参数初始化 (Host)
    std::vector<double> h_weights_ih(INPUT_DIM * HIDDEN_DIM);
    std::vector<double> h_bias_ih(HIDDEN_DIM);
    std::vector<double> h_weights_ho(HIDDEN_DIM * OUTPUT_DIM);
    std::vector<double> h_bias_ho(OUTPUT_DIM);

    // Initialize weights (e.g., Xavier/Glorot initialization or simple random)
    // For simplicity, using small random values
    srand(static_cast<unsigned>(time(0)));
    auto init_weights = [](std::vector<double>& vec, int in_dim, int out_dim) {
        double limit = sqrt(6.0 / (in_dim + out_dim));
        for (auto& val : vec) {
            val = ((double)rand() / RAND_MAX) * 2 * limit - limit;
        }
    };
    auto init_bias = [](std::vector<double>& vec) {
        for (auto& val : vec) {
            val = 0.01; // Small constant bias
        }
    };

    init_weights(h_weights_ih, INPUT_DIM, HIDDEN_DIM);
    init_bias(h_bias_ih);
    init_weights(h_weights_ho, HIDDEN_DIM, OUTPUT_DIM);
    init_bias(h_bias_ho);
    std::cout << "[INFO] MLP weights and biases initialized." << std::endl;

    // Device memory allocation
    double *d_train_X, *d_train_y, *d_test_X, *d_test_y;
    double *d_weights_ih, *d_bias_ih, *d_weights_ho, *d_bias_ho;
    double *d_grad_weights_ih, *d_grad_bias_ih, *d_grad_weights_ho, *d_grad_bias_ho;
    double *d_hidden_activations, *d_hidden_linear, *d_output, *d_loss_array;
    double *d_delta_output, *d_delta_hidden;

    HIP_CHECK(hipMalloc(&d_train_X, train_samples * INPUT_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_train_y, train_samples * OUTPUT_DIM * sizeof(double))); // Assuming OUTPUT_DIM for y as well
    HIP_CHECK(hipMalloc(&d_test_X, test_samples * INPUT_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_test_y, test_samples * OUTPUT_DIM * sizeof(double)));

    HIP_CHECK(hipMalloc(&d_weights_ih, INPUT_DIM * HIDDEN_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_bias_ih, HIDDEN_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_weights_ho, HIDDEN_DIM * OUTPUT_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_bias_ho, OUTPUT_DIM * sizeof(double)));

    HIP_CHECK(hipMalloc(&d_grad_weights_ih, INPUT_DIM * HIDDEN_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_grad_bias_ih, HIDDEN_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_grad_weights_ho, HIDDEN_DIM * OUTPUT_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_grad_bias_ho, OUTPUT_DIM * sizeof(double)));

    HIP_CHECK(hipMalloc(&d_hidden_linear, BATCH_SIZE * HIDDEN_DIM * sizeof(double))); // For z of hidden layer
    HIP_CHECK(hipMalloc(&d_hidden_activations, BATCH_SIZE * HIDDEN_DIM * sizeof(double))); // For a of hidden layer (after ReLU)
    HIP_CHECK(hipMalloc(&d_output, BATCH_SIZE * OUTPUT_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_loss_array, BATCH_SIZE * OUTPUT_DIM * sizeof(double))); // Per-sample loss

    HIP_CHECK(hipMalloc(&d_delta_output, BATCH_SIZE * OUTPUT_DIM * sizeof(double))); // Gradient of loss w.r.t output layer's linear output
    HIP_CHECK(hipMalloc(&d_delta_hidden, BATCH_SIZE * HIDDEN_DIM * sizeof(double))); // Gradient of loss w.r.t hidden layer's linear output (before ReLU)

    // Copy initial weights and biases to device
    HIP_CHECK(hipMemcpy(d_weights_ih, h_weights_ih.data(), INPUT_DIM * HIDDEN_DIM * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bias_ih, h_bias_ih.data(), HIDDEN_DIM * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weights_ho, h_weights_ho.data(), HIDDEN_DIM * OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bias_ho, h_bias_ho.data(), OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice));

    std::cout << "[INFO] Device memory allocated and initial parameters copied." << std::endl;

        // 训练MLP网络，包括前向传播、反向传播、梯度下降、参数更新等
    dim3 threadsPerBlock(16, 16); // Example, tune as needed
    // For bias kernels or 1D kernels
    dim3 threadsPerBlock_1D(256);

    std::cout << "[INFO] Starting training..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        double total_epoch_loss = 0.0;
        int num_batches = (train_samples + BATCH_SIZE - 1) / BATCH_SIZE;

        // Shuffle training data (optional, but good practice)
        // For simplicity, not implemented here. Usually, you'd shuffle indices of h_train_X and h_train_y.

        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            int current_batch_size = BATCH_SIZE;
            int offset = batch_idx * BATCH_SIZE;
            if (batch_idx == num_batches - 1) { // Last batch might be smaller
                current_batch_size = train_samples - offset;
            }
            if (current_batch_size <= 0) continue;

            // Device pointers for current batch (can reuse d_hidden_*, d_output etc. if sized for max BATCH_SIZE)
            double *d_batch_X, *d_batch_y;
            HIP_CHECK(hipMalloc(&d_batch_X, current_batch_size * INPUT_DIM * sizeof(double)));
            HIP_CHECK(hipMalloc(&d_batch_y, current_batch_size * OUTPUT_DIM * sizeof(double)));

            HIP_CHECK(hipMemcpy(d_batch_X, h_train_X.data() + offset * INPUT_DIM, current_batch_size * INPUT_DIM * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_batch_y, h_train_y.data() + offset * OUTPUT_DIM, current_batch_size * OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice));

            // --- Forward Pass ---
            // Hidden Layer: Z1 = X * W_ih
            dim3 gridDim_ih_matmul((HIDDEN_DIM + threadsPerBlock.x -1) / threadsPerBlock.x, 
                                   (current_batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
            matmul<<<gridDim_ih_matmul, threadsPerBlock>>>(d_batch_X, d_weights_ih, d_hidden_linear, current_batch_size, HIDDEN_DIM, INPUT_DIM);
            HIP_CHECK(hipGetLastError());
            
            // Hidden Layer: Z1 += b_ih
            dim3 gridDim_ih_bias_add((HIDDEN_DIM + threadsPerBlock.x -1) / threadsPerBlock.x, 
                                     (current_batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
            add_bias<<<gridDim_ih_bias_add, threadsPerBlock>>>(d_hidden_linear, d_bias_ih, current_batch_size, HIDDEN_DIM);
            HIP_CHECK(hipGetLastError());

            // Hidden Layer: A1 = relu(Z1)
            dim3 gridDim_relu_hidden(((current_batch_size * HIDDEN_DIM) + threadsPerBlock_1D.x -1) / threadsPerBlock_1D.x);
            HIP_CHECK(hipMemcpy(d_hidden_activations, d_hidden_linear, current_batch_size * HIDDEN_DIM * sizeof(double), hipMemcpyDeviceToDevice)); // copy Z1 to A1 before relu
            relu_forward<<<gridDim_relu_hidden, threadsPerBlock_1D>>>(d_hidden_activations, current_batch_size * HIDDEN_DIM);
            HIP_CHECK(hipGetLastError());

            // Output Layer: Z2 = A1 * W_ho
            dim3 gridDim_ho_matmul((OUTPUT_DIM + threadsPerBlock.x -1) / threadsPerBlock.x, 
                                   (current_batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
            matmul<<<gridDim_ho_matmul, threadsPerBlock>>>(d_hidden_activations, d_weights_ho, d_output, current_batch_size, OUTPUT_DIM, HIDDEN_DIM);
            HIP_CHECK(hipGetLastError());

            // Output Layer: Z2 += b_ho
            dim3 gridDim_ho_bias_add((OUTPUT_DIM + threadsPerBlock.x -1) / threadsPerBlock.x, 
                                     (current_batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
            add_bias<<<gridDim_ho_bias_add, threadsPerBlock>>>(d_output, d_bias_ho, current_batch_size, OUTPUT_DIM);
            HIP_CHECK(hipGetLastError());
            // d_output now contains final predictions for the batch

            // --- Loss Calculation ---
            dim3 gridDim_loss(((current_batch_size * OUTPUT_DIM) + threadsPerBlock_1D.x -1) / threadsPerBlock_1D.x);
            compute_mse_loss<<<gridDim_loss, threadsPerBlock_1D>>>(d_output, d_batch_y, d_loss_array, current_batch_size * OUTPUT_DIM);
            HIP_CHECK(hipGetLastError());
            
            std::vector<double> h_batch_loss(current_batch_size * OUTPUT_DIM);
            HIP_CHECK(hipMemcpy(h_batch_loss.data(), d_loss_array, current_batch_size * OUTPUT_DIM * sizeof(double), hipMemcpyDeviceToHost));
            double current_batch_total_loss = 0;
            for(double l : h_batch_loss) total_epoch_loss += l; // Accumulate sum of squared errors
            current_batch_total_loss = total_epoch_loss; // just for this batch, for potential per-batch print

            // --- Backward Pass ---
            // Gradient of loss w.r.t output layer's linear output (Z2)
            // For MSE: dL/dZ2 = (A2 - Y)
            // compute_output_grad computes (pred - target), which is dL/dZ2 if pred=Z2
            dim3 gridDim_grad_output(((current_batch_size * OUTPUT_DIM) + threadsPerBlock_1D.x -1) / threadsPerBlock_1D.x);
            compute_output_grad<<<gridDim_grad_output, threadsPerBlock_1D>>>(d_output, d_batch_y, d_delta_output, current_batch_size * OUTPUT_DIM);
            HIP_CHECK(hipGetLastError());

            // Gradient for W_ho: dL/dW_ho = A1^T * dL/dZ2
            dim3 gridDim_grad_Who((OUTPUT_DIM + threadsPerBlock.x -1) / threadsPerBlock.x, 
                                  (HIDDEN_DIM + threadsPerBlock.y -1) / threadsPerBlock.y);
            // matmul_backward_weights(upstream_grad, layer_input, weight_grad, M_upstream, N_layer_input_features, K_upstream_features_==_output_dim_of_current_layer)
            // M_upstream = current_batch_size
            // N_layer_input_features = HIDDEN_DIM (features of A1)
            // K_upstream_features = OUTPUT_DIM (features of dL/dZ2)
            // weight_grad is (HIDDEN_DIM x OUTPUT_DIM)
            matmul_backward_weights<<<gridDim_grad_Who, threadsPerBlock>>>(d_delta_output, d_hidden_activations, d_grad_weights_ho, current_batch_size, HIDDEN_DIM, OUTPUT_DIM);
            HIP_CHECK(hipGetLastError());

            // Gradient for b_ho: dL/db_ho = sum(dL/dZ2, axis=0) / batch_size
            dim3 gridDim_grad_bho((OUTPUT_DIM + threadsPerBlock_1D.x - 1) / threadsPerBlock_1D.x);
            sum_gradients_for_bias<<<gridDim_grad_bho, threadsPerBlock_1D>>>(d_delta_output, d_grad_bias_ho, current_batch_size, OUTPUT_DIM);
            HIP_CHECK(hipGetLastError());

            // Gradient w.r.t hidden layer activations A1: dL/dA1 = dL/dZ2 * W_ho^T
            // d_delta_hidden will store dL/dA1 temporarily
            dim3 gridDim_grad_A1((HIDDEN_DIM + threadsPerBlock.x -1) / threadsPerBlock.x, 
                                 (current_batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
            // matmul_backward_input(upstream_grad, weights, input_grad, M_upstream, N_input_features, K_upstream_features_==_weight_output_dim)
            // M_upstream = current_batch_size
            // N_input_features = HIDDEN_DIM (features of A1 / Z1)
            // K_upstream_features = OUTPUT_DIM (features of dL/dZ2, also output dim of W_ho)
            // weights = W_ho (HIDDEN_DIM x OUTPUT_DIM)
            // input_grad = dL/dA1 (current_batch_size x HIDDEN_DIM) -> stored in d_delta_hidden
            matmul_backward_input<<<gridDim_grad_A1, threadsPerBlock>>>(d_delta_output, d_weights_ho, d_delta_hidden, current_batch_size, HIDDEN_DIM, OUTPUT_DIM);
            HIP_CHECK(hipGetLastError());

            // Gradient w.r.t hidden layer linear output Z1: dL/dZ1 = dL/dA1 * g'(Z1)
            // g'(Z1) is 1 if Z1 > 0, 0 otherwise. compute_relu_backward modifies dL/dA1 in place to get dL/dZ1.
            // d_delta_hidden currently holds dL/dA1. d_hidden_linear holds Z1.
            dim3 gridDim_relu_back(((current_batch_size * HIDDEN_DIM) + threadsPerBlock_1D.x -1) / threadsPerBlock_1D.x);
            compute_relu_backward<<<gridDim_relu_back, threadsPerBlock_1D>>>(d_delta_hidden, d_hidden_linear, current_batch_size * HIDDEN_DIM);
            HIP_CHECK(hipGetLastError()); // d_delta_hidden now has dL/dZ1

            // Gradient for W_ih: dL/dW_ih = X^T * dL/dZ1
            dim3 gridDim_grad_Wih((HIDDEN_DIM + threadsPerBlock.x -1) / threadsPerBlock.x, 
                                  (INPUT_DIM + threadsPerBlock.y -1) / threadsPerBlock.y);
            // M_upstream = current_batch_size
            // N_layer_input_features = INPUT_DIM (features of X)
            // K_upstream_features = HIDDEN_DIM (features of dL/dZ1)
            // weight_grad is (INPUT_DIM x HIDDEN_DIM)
            matmul_backward_weights<<<gridDim_grad_Wih, threadsPerBlock>>>(d_delta_hidden, d_batch_X, d_grad_weights_ih, current_batch_size, INPUT_DIM, HIDDEN_DIM);
            HIP_CHECK(hipGetLastError());

            // Gradient for b_ih: dL/db_ih = sum(dL/dZ1, axis=0) / batch_size
            dim3 gridDim_grad_bih((HIDDEN_DIM + threadsPerBlock_1D.x - 1) / threadsPerBlock_1D.x);
            sum_gradients_for_bias<<<gridDim_grad_bih, threadsPerBlock_1D>>>(d_delta_hidden, d_grad_bias_ih, current_batch_size, HIDDEN_DIM);
            HIP_CHECK(hipGetLastError());

            // --- Parameter Update (SGD) ---
            sgd_update<<<gridDim_grad_Wih, threadsPerBlock>>>(d_weights_ih, d_grad_weights_ih, LEARNING_RATE, INPUT_DIM * HIDDEN_DIM);
            sgd_update<<<gridDim_grad_bih, threadsPerBlock_1D>>>(d_bias_ih, d_grad_bias_ih, LEARNING_RATE, HIDDEN_DIM);
            sgd_update<<<gridDim_grad_Who, threadsPerBlock>>>(d_weights_ho, d_grad_weights_ho, LEARNING_RATE, HIDDEN_DIM * OUTPUT_DIM);
            sgd_update<<<gridDim_grad_bho, threadsPerBlock_1D>>>(d_bias_ho, d_grad_bias_ho, LEARNING_RATE, OUTPUT_DIM);
            HIP_CHECK(hipGetLastError());

            HIP_CHECK(hipFree(d_batch_X));
            HIP_CHECK(hipFree(d_batch_y));
        }
        
        double average_epoch_loss = total_epoch_loss / (train_samples * OUTPUT_DIM) ; // Average MSE per sample component
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
        std::cout << "[Epoch " << epoch + 1 << "/" << EPOCHS << "] Loss: " << average_epoch_loss 
                  << ", Time: " << elapsed_ms << " ms" << std::endl;
        start_time = current_time; // Reset timer for next epoch or use total time from beginning
    }
    std::cout << "[INFO] Training finished." << std::endl;

        // 推理部分，测试训练的MLP网络
    std::cout << "\n[INFO] Starting inference on test set..." << std::endl;
    std::vector<double> h_predictions(test_samples * OUTPUT_DIM);
    double *d_batch_test_X, *d_batch_test_y; // Reused for inference batches if needed, or single full pass

    // For simplicity, let's do inference in one go if test_samples is not too large.
    // Otherwise, batching similar to training is needed.
    // Assuming test_samples fits BATCH_SIZE for intermediate buffers, or adjust buffers.
    // For a true batched inference, the loop structure would be similar to training but without backprop.

    // Ensure device buffers for activations are large enough for test_samples if doing single pass
    // Or, re-allocate if test_samples > BATCH_SIZE (and we used BATCH_SIZE fixed allocation for d_hidden_linear etc.)
    // For now, assume BATCH_SIZE was a max, and test_samples can be processed if <= BATCH_SIZE for those intermediate buffers.
    // A robust solution would re-allocate d_hidden_linear, d_hidden_activations, d_output for test_samples size if needed.

    HIP_CHECK(hipMemcpy(d_test_X, h_test_X.data(), test_samples * INPUT_DIM * sizeof(double), hipMemcpyHostToDevice));
    // d_test_y is not strictly needed for prediction output but for calculating final MSE on test set.
    HIP_CHECK(hipMemcpy(d_test_y, h_test_y.data(), test_samples * OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice));

    // Forward pass for test set (using test_samples as batch size here)
    // Adjust d_hidden_linear, d_hidden_activations, d_output, d_loss_array if their size is BATCH_SIZE and test_samples > BATCH_SIZE
    // For this example, let's assume they are large enough or test_samples <= BATCH_SIZE.
    // If not, they need to be reallocated or inference needs to be batched.
    // Let's re-purpose the existing d_hidden_linear, d_hidden_activations, d_output, d_loss_array 
    // by ensuring they are either allocated to max(BATCH_SIZE, test_samples) or batched inference.
    // Assuming test_samples will be processed. The device arrays for activations might need to be test_samples large.
    // For simplicity in this step, if test_samples > BATCH_SIZE, this will error with current fixed-size (BATCH_SIZE) buffers.
    // A proper implementation would handle this with batched inference or larger buffers.

    if (test_samples > 0) {
        // Re-allocate intermediate buffers if test_samples > BATCH_SIZE
        // These buffers are d_hidden_linear, d_hidden_activations, d_output, d_loss_array
        if (test_samples > BATCH_SIZE) {
            std::cout << "[WARN] Test samples (" << test_samples << ") > BATCH_SIZE (" << BATCH_SIZE 
                      << "). Reallocating intermediate buffers for inference." << std::endl;
            
            // Free the old, smaller buffers
            HIP_CHECK(hipFree(d_hidden_linear)); 
            HIP_CHECK(hipFree(d_hidden_activations)); 
            HIP_CHECK(hipFree(d_output)); 
            HIP_CHECK(hipFree(d_loss_array));

            // Allocate new, larger buffers using the original pointers
            HIP_CHECK(hipMalloc(&d_hidden_linear, test_samples * HIDDEN_DIM * sizeof(double)));
            HIP_CHECK(hipMalloc(&d_hidden_activations, test_samples * HIDDEN_DIM * sizeof(double)));
            HIP_CHECK(hipMalloc(&d_output, test_samples * OUTPUT_DIM * sizeof(double)));
            HIP_CHECK(hipMalloc(&d_loss_array, test_samples * OUTPUT_DIM * sizeof(double)));
        }

        // Now d_hidden_linear, d_hidden_activations, d_output, d_loss_array are guaranteed to be large enough
        // either because test_samples <= BATCH_SIZE or they were just reallocated.

        dim3 gridDim_ih_matmul_test((HIDDEN_DIM + threadsPerBlock.x -1) / threadsPerBlock.x, 
                                   (test_samples + threadsPerBlock.y - 1) / threadsPerBlock.y);
        matmul<<<gridDim_ih_matmul_test, threadsPerBlock>>>(d_test_X, d_weights_ih, d_hidden_linear, test_samples, HIDDEN_DIM, INPUT_DIM); // Use d_hidden_linear
        HIP_CHECK(hipGetLastError());
        
        dim3 gridDim_ih_bias_add_test((HIDDEN_DIM + threadsPerBlock.x -1) / threadsPerBlock.x, 
                                     (test_samples + threadsPerBlock.y - 1) / threadsPerBlock.y);
        add_bias<<<gridDim_ih_bias_add_test, threadsPerBlock>>>(d_hidden_linear, d_bias_ih, test_samples, HIDDEN_DIM); // Use d_hidden_linear
        HIP_CHECK(hipGetLastError());

        dim3 gridDim_relu_hidden_test(((test_samples * HIDDEN_DIM) + threadsPerBlock_1D.x -1) / threadsPerBlock_1D.x);
        HIP_CHECK(hipMemcpy(d_hidden_activations, d_hidden_linear, test_samples * HIDDEN_DIM * sizeof(double), hipMemcpyDeviceToDevice)); // Use d_hidden_activations, d_hidden_linear
        relu_forward<<<gridDim_relu_hidden_test, threadsPerBlock_1D>>>(d_hidden_activations, test_samples * HIDDEN_DIM); // Use d_hidden_activations
        HIP_CHECK(hipGetLastError());

        dim3 gridDim_ho_matmul_test((OUTPUT_DIM + threadsPerBlock.x -1) / threadsPerBlock.x, 
                                   (test_samples + threadsPerBlock.y - 1) / threadsPerBlock.y);
        matmul<<<gridDim_ho_matmul_test, threadsPerBlock>>>(d_hidden_activations, d_weights_ho, d_output, test_samples, OUTPUT_DIM, HIDDEN_DIM); // Use d_hidden_activations, d_output
        HIP_CHECK(hipGetLastError());

        dim3 gridDim_ho_bias_add_test((OUTPUT_DIM + threadsPerBlock.x -1) / threadsPerBlock.x, 
                                     (test_samples + threadsPerBlock.y - 1) / threadsPerBlock.y);
        add_bias<<<gridDim_ho_bias_add_test, threadsPerBlock>>>(d_output, d_bias_ho, test_samples, OUTPUT_DIM); // Use d_output
        HIP_CHECK(hipGetLastError());

        HIP_CHECK(hipMemcpy(h_predictions.data(), d_output, test_samples * OUTPUT_DIM * sizeof(double), hipMemcpyDeviceToHost)); // Use d_output
        
        // Calculate MSE on test set
        dim3 gridDim_loss_test(((test_samples * OUTPUT_DIM) + threadsPerBlock_1D.x -1) / threadsPerBlock_1D.x);
        compute_mse_loss<<<gridDim_loss_test, threadsPerBlock_1D>>>(d_output, d_test_y, d_loss_array, test_samples * OUTPUT_DIM); // Use d_output, d_loss_array
        std::vector<double> h_test_loss(test_samples * OUTPUT_DIM);
        HIP_CHECK(hipMemcpy(h_test_loss.data(), d_loss_array, test_samples * OUTPUT_DIM * sizeof(double), hipMemcpyDeviceToHost)); // Use d_loss_array
        double total_test_mse = 0;
        for(double l : h_test_loss) total_test_mse += l;
        double average_test_mse = total_test_mse / (test_samples * OUTPUT_DIM);
        std::cout << "[INFO] Test MSE: " << average_test_mse << std::endl;

        // Denormalize predictions and actual values for comparison
        std::vector<double> denorm_predictions = h_predictions;
        denormalize_data(denorm_predictions, min_val, max_val);
        std::vector<double> denorm_actual_y = h_test_y; // Original test y was normalized
        denormalize_data(denorm_actual_y, min_val, max_val);

        std::cout << "[INFO] Sample Predictions (denormalized):" << std::endl;
        for (size_t i = 0; i < std::min((size_t)10, test_samples); ++i) {
            std::cout << "  Predicted: " << denorm_predictions[i * OUTPUT_DIM] 
                      << ", Actual: " << denorm_actual_y[i * OUTPUT_DIM] << std::endl;
        }
        // No need for the 'if (reallocated)' block to free d_test_* pointers anymore,
        // as we are directly using and potentially reallocating the main d_hidden_linear etc.
    }

    // Cleanup device memory
    HIP_CHECK(hipFree(d_train_X));
    HIP_CHECK(hipFree(d_train_y));
    HIP_CHECK(hipFree(d_test_X));
    HIP_CHECK(hipFree(d_test_y));
    HIP_CHECK(hipFree(d_weights_ih));
    HIP_CHECK(hipFree(d_bias_ih));
    HIP_CHECK(hipFree(d_weights_ho));
    HIP_CHECK(hipFree(d_bias_ho));
    HIP_CHECK(hipFree(d_grad_weights_ih));
    HIP_CHECK(hipFree(d_grad_bias_ih));
    HIP_CHECK(hipFree(d_grad_weights_ho));
    HIP_CHECK(hipFree(d_grad_bias_ho));
    HIP_CHECK(hipFree(d_hidden_activations));
    HIP_CHECK(hipFree(d_hidden_linear));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_loss_array));
    HIP_CHECK(hipFree(d_delta_output));
    HIP_CHECK(hipFree(d_delta_hidden));

    std::cout << "[INFO] Device memory freed." << std::endl;
    return 0;
}