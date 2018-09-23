
#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"

#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/layers/depthwise_convolution.hpp"

namespace caffe {
	template <typename Dtype>
	void DepthwiseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {		
		ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
		const int* kernel_shape_data = kernel_shape_.cpu_data();
		const int* stride_data = stride_.cpu_data();
		const int* pad_data = pad_.cpu_data();
		args.filter_height = kernel_shape_data[0];
		args.filter_width = kernel_shape_data[1];
		args.stride_height = stride_data[0];
		args.stride_width = stride_data[1];
		args.pad_height = pad_data[0];
		args.pad_width = pad_data[1];
		args.batch = bottom[0]->num();
		args.in_channel = bottom[0]->channels();
		args.in_height = bottom[0]->height();
		args.in_width = bottom[0]->width();
		args.out_channel = this->layer_param_.convolution_param().num_output();

		args.out_height = output_shape_[0];
		args.out_width = output_shape_[1];

	}

	template <typename Dtype>
	void DepthwiseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		ConvolutionLayer<Dtype>::Reshape(bottom, top);
	}

	INSTANTIATE_CLASS(DepthwiseConvolutionLayer);
	REGISTER_LAYER_CLASS(DepthwiseConvolution);
}