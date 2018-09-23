#ifndef CAFFE_DEPTHWISE_CONV_HPP
#define CAFFE_DEPTHWISE_CONV_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/conv_layer.hpp"

namespace caffe {

	typedef struct DepthwiseArgs {
		// Input layer dimensions
		int batch;
		int in_height;
		int in_width;
		int in_channel;
		int filter_height;
		int filter_width;
		int stride_height;
		int stride_width;
		int pad_height;
		int pad_width;

		// Output layer dimensions
		int out_height;
		int out_width;
		int out_channel;
	}DepthwiseArgs;

template <typename Dtype>
class DepthwiseConvolutionLayer : public ConvolutionLayer<Dtype> {
public:
	explicit DepthwiseConvolutionLayer(const LayerParameter& param)
		: ConvolutionLayer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {}
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual inline bool reverse_dimensions() { return false; }

protected:
	DepthwiseArgs args;

};

}  // namespace caffe

#endif // !CAFFE_DEPTHWISE_CONV_HPP

