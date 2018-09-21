#ifndef CAFFE_CUDNN_BATCH_NORM_LAYER_HPP_
#define CAFFE_CUDNN_BATCH_NORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/batch_norm_layer.hpp"

#ifdef USE_CUDNN

namespace caffe {
	template <typename Dtype>
	class CuDNNBatchNormLayer : public BatchNormLayer<Dtype> {
	public:
		explicit CuDNNBatchNormLayer(const LayerParameter& param)
			: BatchNormLayer<Dtype>(param), handles_setup_(false) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual ~CuDNNBatchNormLayer();

	protected:
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		cudnnTensorDescriptor_t bottom_desc_, top_desc_;
		//cudnnTensorDescriptor_t map_desc_;
		cudnnTensorDescriptor_t scale_bias_mean_var_desc_;
		cudnnBatchNormMode_t mode_;

		bool handles_setup_;
		cudnnHandle_t handle_;
		cudaStream_t  stream_;

		Blob<Dtype> save_mean_;
		Blob<Dtype> save_inv_var_;

		Blob<Dtype> scale_ones_;
		Blob<Dtype> bias_zeros_;

	};

}



#endif // USE_CUDNN


#endif