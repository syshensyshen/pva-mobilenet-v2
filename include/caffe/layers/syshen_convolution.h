#ifndef __SYSHEN_CUDA_TEST_HEADER__
#define __SYSHEN_CUDA_TEST_HEADER__
#include<process.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cudnn.h>

#define CHECK_CUDA_ERROR(argument_t) {                           \
    cudaError_t error_t = argument_t;                              \
    if (error_t != cudaSuccess) {                                  \
        printf("Error: %s: %d, ", __FILE__, __LINE__);         \
        printf("code: %d, reason: %s\r\n", error_t, cudaGetErrorString(error_t)); \
        exit(1);                                                 \
	}                                                            \
}

#define CHECK_CUDNN_ERROR(argument_t) {                           \
    cudnnStatus_t error_t = argument_t;                              \
    if (error_t != CUDNN_STATUS_SUCCESS) {                                  \
        printf("Error: %s: %d, ", __FILE__, __LINE__);         \
        printf("code: %d, reason: %s\r\n", error_t, cudnnGetErrorString(error_t)); \
        exit(1);                                                 \
	}                                                            \
}

__global__ void checkIndex();

void test_check();

void test_change_block();

template <typename Dtype>
class syshen_convolution {
public:
	syshen_convolution();
	~syshen_convolution();
	void SetUp();
	
	inline void setInputParam(int batch, int channels, int height, int width) {
		batch = batch;
		in_channels = channels;
		in_h = height;
		in_w = width;
	}

	inline void setInputKernelParam(int stride_h, int stride_w, int pad_h, int pad_w,
		int dilation_h, int dilation_w, int kernel_h, int kernel_w) {
		stride_h = stride_h;
		stride_w = stride_w;
		pad_h = pad_h;
		pad_w = pad_w;
		dilation_h = dilation_h;
		dilation_w = dilation_w;
		kernel_h = kernel_h;
		kernel_w = kernel_w;
	}

	inline void setOutputParam(int out_batch, int out_channels, int out_h, int out_w) {
		out_batch = output_batch;
		out_channels = out_channels;
		out_h = out_h;
		out_w = out_w;
	}

	void Forward(Dtype *input, Dtype *output, Dtype *weights, Dtype *bias_weights) {
		Dtype conv_alpha = 1.0f;
		Dtype conv_beta = 0;
		cudnnConvolutionForward(handle_t, cudnn::dataType<Dtype>::one, input_desc, input,
			filter_dsec, weights, conv_desc, algo, workSpace, workSpaceSize, cudnn::dataType<Dtype>::zero, output_desc, output);
		if (has_bias) {
			cudnnAddTensor(handle_t, cudnn::dataType<Dtype>::one, bias, bias_weights, cudnn::dataType<Dtype>::one, output_desc, output);
		}
	}
protected:

	inline void setStride(int stride_h, int stride_w) {
		stride_h = stride_h;
		stride_w = stride_w;
	}
	inline void setPad(int pad_h, int pad_w) {
		pad_h = pad_h;
		pad_w = pad_w;
	}
	inline void setDliation(int dilation_h, int dilation_w) {
		dilation_h = dilation_h;
		dilation_w = dilation_w;
	}

	inline void setKernel(int kernel_h, int jernel_w) {
		kernel_h = kernel_h;
		kernel_w = kernel_w;
	}
private:
	cudnnTensorDescriptor_t input_desc, output_desc, filter_dsec, conv_desc, bias;
	cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
	cudnnHandle_t handle_t;
	cudaStream_t stream;
	cudaEvent_t start;
	size_t workSpaceSize;
	bool use_stream, has_bias;
	void* workSpace;
	int stride_h, stride_w, pad_h, pad_w;
	int dilation_h, dilation_w, kernel_h, kernel_w;
	int batch, in_channels, in_h, in_w;
	int out_batch, out_channels, out_h, out_w;
};


#endif
