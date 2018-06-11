#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cublas_v2.h>

#include "error_helper.hpp"

#define CUDA_CHECK_ERROR

#define CudaSafeCall(err) __CudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError() __CudaCheckError(__FILE__, __LINE__)

__host__ void __CudaSafeCall(cudaError err, const char *file, const int line) {
#ifdef CUDA_CHECK_ERROR
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line,
                cudaGetErrorString(err));
        exit(-1);
    }
#endif
}

__host__ void __CudaCheckError(const char *file, const int line) {
#ifdef CUDA_CHECK_ERROR
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line,
                cudaGetErrorString(err));
        exit(-1);
    }
#endif
}

// weights & bias size: (filter size * channels + 1 bias) * #filters
const float conv1_1_w = (3 * 3 * 3    + 1) * 64;
const float conv1_2_w = (3 * 3 * 64   + 1) * 64;
const float conv2_1_w = (3 * 3 * 64   + 1) * 128;
const float conv2_2_w = (3 * 3 * 128  + 1) * 128;
const float conv3_1_w = (3 * 3 * 128  + 1) * 256;
const float conv3_2_w = (3 * 3 * 256  + 1) * 256;
const float conv3_3_w = (3 * 3 * 256  + 1) * 256;
const float conv3_4_w = (3 * 3 * 256  + 1) * 256;
const float conv4_1_w = (3 * 3 * 256  + 1) * 512;
const float conv4_2_w = (3 * 3 * 512  + 1) * 512;
const float conv4_3_w = (3 * 3 * 512  + 1) * 512;
const float conv4_4_w = (3 * 3 * 512  + 1) * 512;
const float conv5_1_w = (3 * 3 * 512  + 1) * 512;
const float conv5_2_w = (3 * 3 * 512  + 1) * 512;
const float conv5_3_w = (3 * 3 * 512  + 1) * 512;
const float conv5_4_w = (3 * 3 * 512  + 1) * 512;
const float fc1_w     = (7 * 7 * 512  + 1) * 4096;
const float fc2_w     = (1 * 1 * 4096 + 1) * 4096;
const float fc3_w     = (1 * 1 * 4096 + 1) * 1000;
// layer output size
const float conv1_1  = 224 * 224 * 64;
const float conv1_2  = 224 * 224 * 64;
const float maxpool1 = 112 * 112 * 64;
const float conv2_1  = 112 * 112 * 128;
const float conv2_2  = 112 * 112 * 128;
const float maxpool2 = 56  * 56  * 128;
const float conv3_1  = 56  * 56  * 256;
const float conv3_2  = 56  * 56  * 256;
const float conv3_3  = 56  * 56  * 256;
const float conv3_4  = 56  * 56  * 256;
const float maxpool3 = 28  * 28  * 256;
const float conv4_1  = 28  * 28  * 512;
const float conv4_2  = 28  * 28  * 512;
const float conv4_3  = 28  * 28  * 512;
const float conv4_4  = 28  * 28  * 512;
const float maxpool4 = 14  * 14  * 512;
const float conv5_1  = 14  * 14  * 512;
const float conv5_2  = 14  * 14  * 512;
const float conv5_3  = 14  * 14  * 512;
const float conv5_4  = 14  * 14  * 512;
const float maxpool5 = 7   * 7   * 512;
const float fc1      = 1   * 1   * 4096;
const float fc2      = 1   * 1   * 4096;
const float fc3      = 1   * 1   * 1000;

FILE *fw;
FILE *fb;
cublasHandle_t cubHandle;
// for cublas dummy constant
const float alpha = 1.0f;
const float beta = 0.0f;

// required to normalize by mean pixel (in rgb order)
float mean_pixel[3] = {123.68, 116.779, 103.939};
// input image
float image[224 * 224 * 3];
// ouput of each layer, device pointer
float *d_output;

__global__ void maxpooling(float *output, const float *input, const int width, const int channels)
{
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int new_width = width / 2;
    int i = thread_id / new_width * 2;
    int j = thread_id % new_width * 2;
    int index = i * width + j;

    for (int c = 0; c < channels; c++) {
        float max = 0;
        if (max < input[index * channels + c])
            max = input[index * channels + c];
        if (max < input[(index + 1) * channels + c])
            max = input[(index + 1) * channels + c];
        if (max < input[(index + width) * channels + c])
            max = input[(index + width) * channels + c];
        if (max < input[(index + width + 1) * channels + c])
            max = input[(index + width + 1) * channels + c];
        output[thread_id * channels + c] = max;
    }
}

__global__ void transform_image(float *input, const float *raw_input, const int width, const int channels)
{
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int start_i = thread_id / width - 1;
    int start_j = thread_id % width - 1;
    int per_channel_width = width * width;
    int hidden_width = 3 * 3 * channels + 1;
    int global_offset = thread_id * hidden_width;

    for (int c = 0; c < channels; c++) {
        int offset = 0;
        for (int i = start_i; i < start_i + 3; i++) {
            if (i < 0 || i == width)
                continue;
            for (int j = start_j; j < start_j + 3; j++) {
                if (j < 0 || j == width)
                    continue;
                input[global_offset + c * 9 + offset] = raw_input[c * per_channel_width + i * width + j];
                offset++;
            }
        }
    }
    input[(thread_id + 1) * hidden_width - 1] = 1;
}

__global__ void transform_fc(float *input, const float *raw_input, const int width, const int channels)
{
    int thread_id = threadIdx.x;
    int size = width * width;

    for (int s = 0; s < size; s++)
        input[thread_id * size + s] = raw_input[s * channels + thread_id];
    if (thread_id == 0)
        input[width * width * channels] = 1;
}

__global__ void transform(float *input, const float *raw_input, const int width, const int channels)
{
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int start_i = thread_id / width - 1;
    int start_j = thread_id % width - 1;
    int hidden_width = 3 * 3 * channels + 1;
    int global_offset = thread_id * hidden_width;

    float relu;
    for (int c = 0; c < channels; c++) {
        int offset = 0;
        for (int i = start_i; i < start_i + 3; i++) {
            if (i < 0 || i == width)
                continue;
            for (int j = start_j; j < start_j + 3; j++) {
                if (j < 0 || j == width)
                    continue;
                relu = raw_input[(i * width + j) * channels + c];
                input[global_offset + c * 9 + offset] = relu < 0 ? 0 : relu;
                offset++;
            }
        }
    }
    input[(thread_id + 1) * hidden_width - 1] = 1;
}

void fully_connected(int width, int channels, int num_filters)
{
    int num_weights = (width * width * channels + 1) * num_filters;
    int filter_size = width * width * channels;
    int hidden_width = filter_size + 1;
    float *weights = (float *)malloc(num_weights * sizeof(float));
    for (int i = 0; i < num_filters; i++) {
        for (int j = 0; j < filter_size; j++)
            fscanf(fw, "%f", &weights[i * hidden_width + j]);
        fscanf(fb, "%f", &weights[i * hidden_width + filter_size]);
    }

    float *d_input;
    size_t input_size = (width * width * channels + 1) * sizeof(float);
    CudaSafeCall(cudaMalloc(&d_input, input_size));
    if (width == 1) {
        // previous output vector (channels * 1), expand to ((channels + 1) * 1) with a 1 at last
        float *output = (float *)malloc((channels + 1) * sizeof(float));
        CudaSafeCall(cudaMemcpy(output, d_output, channels * sizeof(float), cudaMemcpyDeviceToHost));
        output[channels] = 1;
        CudaSafeCall(cudaMemcpy(d_input, output, (channels + 1) * sizeof(float), cudaMemcpyHostToDevice));
        free(output);
    }
    else {
        // only the first fc needs to transform previous output to a vector (width * width * channels)
        transform_fc <<< 1, channels >>> (d_input, d_output, width, channels);
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());
    }

    float *d_weights;
    CudaSafeCall(cudaMalloc(&d_weights, num_weights * sizeof(float)));
    cudaFree(d_output);
    CudaSafeCall(cudaMalloc(&d_output, num_filters * sizeof(float)));
    error_check(cublasSetMatrix(hidden_width, num_filters, sizeof(float), weights, hidden_width, d_weights, hidden_width));
    // weights * input = (num_filters * (channels + 1)) * ((channels + 1) * 1), consider vector as matrix
    error_check(cublasSgemm(cubHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1, num_filters, hidden_width,
                            &alpha, d_input, 1, d_weights, hidden_width,
                            &beta, d_output, 1));

    free(weights);
    cudaFree(d_input);
    cudaFree(d_weights);
}

void maxpool(int width, int channels)
{
    float *d_temp;
    size_t mem_size = width * width * channels * sizeof(float);
    CudaSafeCall(cudaMalloc(&d_temp, mem_size));
    CudaSafeCall(cudaMemcpy(d_temp, d_output, mem_size, cudaMemcpyDeviceToDevice));
    cudaFree(d_output);
    CudaSafeCall(cudaMalloc(&d_output, mem_size / 4));
    maxpooling <<< width / 2, width / 2 >>> (d_output, d_temp, width, channels);
    CudaCheckError();
    CudaSafeCall(cudaDeviceSynchronize());
}

void convolution(int width, int channels, int num_filters)
{
    int num_weights = (3 * 3 * channels + 1) * num_filters;
    int output_size = width * width * num_filters;
    int filter_size = 3 * 3 * channels;
    int hidden_width = 3 * 3 * channels + 1;
    float *weights = (float *)malloc(num_weights * sizeof(float));
    for (int i = 0; i < num_filters; i++) {
        for (int j = 0; j < filter_size; j++)
            fscanf(fw, "%f", &weights[j * num_filters + i]);
        fscanf(fb, "%f", &weights[filter_size * num_filters + i]);
    }

    float *d_raw_input;
    float *d_input;
    size_t input_size = width * width * hidden_width * sizeof(float);
    CudaSafeCall(cudaMalloc(&d_input, input_size));
    CudaSafeCall(cudaMemset(d_input, 0, input_size));
    // expand original input to (width * width) * (3 * 3 * channels + 1) with a 1 at last for bias
    if (channels == 3) {
        size_t raw_input_size = width * width * channels * sizeof(float);
        CudaSafeCall(cudaMalloc(&d_raw_input, raw_input_size));
        CudaSafeCall(cudaMemcpy(d_raw_input, image, raw_input_size, cudaMemcpyHostToDevice));
        transform_image <<< width, width >>> (d_input, d_raw_input, width, channels);
    }
    else 
        transform <<< width, width >>> (d_input, d_output, width, channels);
    CudaCheckError();
    CudaSafeCall(cudaDeviceSynchronize());

    float *d_weights;
    CudaSafeCall(cudaMalloc(&d_weights, num_weights * sizeof(float)));
    cudaFree(d_output);
    CudaSafeCall(cudaMalloc(&d_output, output_size * sizeof(float)));
    error_check(cublasSetMatrix(num_filters, hidden_width, sizeof(float), weights, num_filters, d_weights, num_filters));
    // input * weights = ((width * width) * (3 * 3 * channels + 1)) * ((3 * 3 * channels + 1) * num_filters)
    error_check(cublasSgemm(cubHandle, CUBLAS_OP_N, CUBLAS_OP_N, num_filters, width * width, hidden_width,
                            &alpha, d_weights, num_filters, d_input, hidden_width,
                            &beta, d_output, num_filters));

    free(weights);
    if (channels == 3)
        cudaFree(d_raw_input);
    cudaFree(d_input);
    cudaFree(d_weights);
}

// debug use, print out each element of output after a layer
void debug_print(int width, int channels)
{
    int output_size = width * width * channels;
    float *output = (float *)malloc(output_size * sizeof(float));
    error_check(cublasGetMatrix(num_filters, width * width, sizeof(float), d_output, num_filters, output, num_filters));
    for (int i = 0; i < channels; i++) {
        for (int j = 0; j < width * width; j++)
            printf("%f ", output[j * channels + i]);
        printf("\n");
    }
    free(output);
}

void write_output(char *output_file)
{
    FILE *fout = fopen(output_file, "w");

    float *output = (float *)malloc(1000 * sizeof(float));
    CudaSafeCall(cudaMemcpy(output, d_output, 1000 * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 1000; i++)
        fprintf(fout, "%f\n", output[i]);

    free(output);
    cudaFree(d_output);
    fclose(fout);
}

void read_image(char *image_file)
{
    FILE *fin = fopen(image_file, "r");
    int total = 224 * 224 * 3;
    for (int index = 0; index < total; index++) {
        fscanf(fin, "%f", &image[index]);
        image[index] -= mean_pixel[index / 50176]; // 50176 = 224 * 224
    }
    fclose(fin);
}

int main(int argc, char **argv)
{
    char *image_file = argv[1];
    char *weights_file = argv[2];
    char *bias_file = argv[3];
    char *output_file = argv[4];
    // read image file
    read_image(image_file);

    // initialize
    fw = fopen(weights_file, "r");
    fb = fopen(bias_file, "r");
    error_check(cublasCreate(&cubHandle));

    // ReLU layers in transform kernel or maxpooling
    // read file input in each layer beginning to save memory cost
    convolution(224, 3, 64);
    convolution(224, 64, 64);
    maxpool(224, 64);
    convolution(112, 64, 128);
    convolution(112, 128, 128);
    maxpool(112, 128);
    convolution(56, 128, 256);
    convolution(56, 256, 256);
    convolution(56, 256, 256);
    convolution(56, 256, 256);
    maxpool(56, 256);
    convolution(28, 256, 512);
    convolution(28, 512, 512);
    convolution(28, 512, 512);
    convolution(28, 512, 512);
    maxpool(28, 512);
    convolution(14, 512, 512);
    convolution(14, 512, 512);
    convolution(14, 512, 512);
    convolution(14, 512, 512);
    maxpool(14, 512);
    fully_connected(7, 512, 4096); // most time consuming file input
    fully_connected(1, 4096, 4096);
    fully_connected(1, 4096, 1000);

    // write 1000 dimension
    write_output(output_file);

    fclose(fw);
    fclose(fb);
    error_check(cublasDestroy(cubHandle));

    return 0;
}
