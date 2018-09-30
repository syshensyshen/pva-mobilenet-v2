
#include "caffe/layers/syshen_convolution.h"


template <typename Dtype>
syshen_convolution<Dtype>::syshen_convolution() {
	CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&input_desc));
	CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&output_desc));
	CHECK_CUDNN_ERROR(cudnnCreateFilterDescriptor(&filter_desc));
	CHECK_CUDNN_ERROR(cudnnCreateConvolutionDescriptor(&conv_desc));
	if (has_bias) {
		CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&bias));
	}
	CHECK_CUDNN_ERROR(cudnnCreate(&handle_t));
	if (use_stream) {
		CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
		CHECK_CUDA_ERROR(cudaEventCreate(&start));
	}
	batch = 1;
	in_channels = 1;
	stride_h = 1;
	stride_w = 1;
	pad_h = 1;
	pad_w = 1;
	dilation_h = 1;
	dilation_w = 1;
	kernel_h = 1;
	kernel_w = 1;
}

template <typename Dtype>
syshen_convolution<Dtype>::~syshen_convolution() {
	CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(input_desc));
	CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(output_desc));
	CHECK_CUDNN_ERROR(cudnnDestroyFilterDescriptor(filter_desc));
	CHECK_CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(&conv_desc));
	if (has_bias) {
		CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(bias));
	}
	CHECK_CUDNN_ERROR(cudnnDestroy(&handle_t));
	if (use_stream) {
		CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
		CHECK_CUDA_ERROR(cudaEventDestroy(strat));
	}
}

template<typename Dtype>
void syshen_convolution<Dtype>::SetUp() {
	int nStride = in_channels * in_h * in_w;
	int cStride = in_h * in_w;

	CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptorEx(
		input_desc, 
		cudnnDataType_t::CUDNN_DATA_FLOAT, 
		batch, 
		in_channels,
		in_h, in_w, nStride, cStride, in_w, 1));

	CHECK_CUDNN_ERROR(cudnnSetFilter4dDescriptor(
		filter_dsec,
		cudnnDataType_t::CUDNN_DATA_FLOAT, 
		CUDNN_TENSOR_NCHW, 
		out_channels, in_channels, kernel_h, kernel_w));

	CHECK_CUDNN_ERROR(cudnnSetConvolution2dDescriptor(
		conv_desc, pad_h, pad_w, stride_h, 
		stride_w, dilation_h, dilation_w, 
		CUDNN_CROSS_CORRELATION, cudnnDataType_t::CUDNN_DATA_FLOAT));

	/*CHECK_CUDNN_ERROR(cudnnGetConvolution2dForwardOutputDim(
		conv_desc, input_desc, filter_dsec,
		&out_batch, &out_channels, &out_h, &out_w));*/

	CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(
		output_desc, CUDNN_TENSOR_NCHW,
		cudnnDataType_t::CUDNN_DATA_FLOAT, 
		out_batch, out_channels, out_h, out_w));

	CHECK_CUDNN_ERROR(cudnnGetConvolutionForwardAlgorithm(
		handle_t, input_desc, filter_dsec, 
		conv_desc, output_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0, &algo));

	CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardFilterWorkspaceSize(
		handle_t, input_desc, filter_dsec,
		conv_desc, output_desc, algo, &workSpaceSize));
	if (0 != workSpaceSize)
		CHECK_CUDA_ERROR(cudaMalloc((void**)&workSpace, workSpaceSize));

	if (has_bias) {
		cudnnSetTensor4dDescriptor(bias, CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, out_batch, out_channels, 1, 1);
	}
}