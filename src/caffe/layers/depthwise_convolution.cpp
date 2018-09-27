
#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"

#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/layers/depthwise_convolution.hpp"

namespace caffe {
	
	template <typename Dtype>
    void DepthwiseConvolutionLayer<Dtype>::compute_output_shape() {
      const int* kernel_shape_data = this->kernel_shape_.cpu_data();
      const int* stride_data = this->stride_.cpu_data();
      const int* pad_data = this->pad_.cpu_data();
      const int* dilation_data = this->dilation_.cpu_data();
      this->output_shape_.clear();
      for (int i = 0; i < this->num_spatial_axes_; ++i) {
        // i + 1 to skip channel axis
        const int input_dim = this->input_shape(i + 1);
        const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
        const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
            / stride_data[i] + 1;
        this->output_shape_.push_back(output_dim);
      }
    }
	
	template <typename Dtype>
	void DepthwiseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {		
		BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
		//const int* kernel_shape_data = this->kernel_shape_.cpu_data();
		//const int* stride_data = this->stride_.cpu_data();
		//const int* pad_data = this->pad_.cpu_data();
		//args.filter_height = kernel_shape_data[0];
		//args.filter_width = kernel_shape_data[1];
		//args.stride_height = stride_data[0];
		//args.stride_width = stride_data[1];
		//args.pad_height = pad_data[0];
		//args.pad_width = pad_data[1];
		//args.batch = bottom[0]->num();
		//args.in_channel = bottom[0]->channels();
		//args.in_height = bottom[0]->height();
		//args.in_width = bottom[0]->width();
		//args.out_channel = this->layer_param_.convolution_param().num_output();

		//args.out_height = this->output_shape_[0];
		//args.out_width = this->output_shape_[1];

	}

	template <typename Dtype>
	void DepthwiseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
	}

	INSTANTIATE_CLASS(DepthwiseConvolutionLayer);
#ifndef USE_CUDNN_DEPTHWISE_CONV
	REGISTER_LAYER_CLASS(DepthwiseConvolution);
#endif
}
