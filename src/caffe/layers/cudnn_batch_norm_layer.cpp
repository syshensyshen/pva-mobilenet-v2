#ifdef USE_CUDNN
#include <algorithm>
#include <vector>

#include "caffe/layers/cudnn_batch_norm_layer.hpp"

namespace caffe {
	template <typename Dtype>
	void CuDNNBatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		BatchNormLayer<Dtype>::LayerSetUp(bottom, top);
		
		CUDNN_CHECK(cudnnCreate(&handle_));
		//CUDA_CHECK(cudaStreamCreate(&stream_));
		int N = bottom[0]->num();
		int C = bottom[0]->channels();
		int H = bottom[0]->height();
		int W = bottom[0]->width();
		cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
		cudnn::createTensor4dDesc<Dtype>(&top_desc_);
		cudnn::createTensor4dDesc<Dtype>(&scale_bias_mean_var_desc_);

#if CUDNN_VERSION_MIN(7, 0, 0)
		mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#else
		mode_ = CUDNN_BATCHNORM_SPATIAL;
#endif

		const vector<int> shape{ 1, C, 1, 1 };
		if (!this->scale_bias_) { // stubs for cudnn
			scale_ones_.Reshape(shape);
			caffe_set(scale_ones_.count(), Dtype(1.0), scale_ones_.mutable_cpu_data());
			bias_zeros_.Reshape(shape);
			caffe_set(bias_zeros_.count(), Dtype(1.0), scale_ones_.mutable_cpu_data());
		}
		save_mean_.Reshape(shape);
		save_inv_var_.Reshape(shape);
		handles_setup_ = true;
	}

	template <typename Dtype>
	void CuDNNBatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		BatchNormLayer<Dtype>::Reshape(bottom, top);
		int N = bottom[0]->num();
		int C = bottom[0]->channels();
		int H = bottom[0]->height();
		int W = bottom[0]->width();
		cudnn::setTensor4dDesc<Dtype>(&bottom_desc_, N, C, H, W);
		cudnn::setTensor4dDesc<Dtype>(&top_desc_, N, C, H, W);
		vector<int> shape{ 1, C, 1, 1 };
		save_mean_.Reshape(shape);
		save_inv_var_.Reshape(shape);
		if (!this->scale_bias_) {
			int C_old = scale_ones_.channels();
			if (C_old != C) {
				scale_ones_.Reshape(shape);
				bias_zeros_.Reshape(shape);
				caffe_set(scale_ones_.count(), Dtype(1.0f), scale_ones_.mutable_cpu_data());
				caffe_set(bias_zeros_.count(), Dtype(0.0f), bias_zeros_.mutable_cpu_data());
			}
		}
		CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(scale_bias_mean_var_desc_, bottom_desc_, mode_));
		CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(scale_bias_mean_var_desc_, top_desc_, mode_));
		

	}

	template <typename Dtype>
	CuDNNBatchNormLayer<Dtype>::~CuDNNBatchNormLayer() {
		if (handles_setup_) {
			CUDNN_CHECK(cudnnDestroy(handle_));
			//CUDA_CHECK(cudaStreamDestroy(&stream));
		}
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_desc_));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(scale_bias_mean_var_desc_));
	}

	INSTANTIATE_CLASS(CuDNNBatchNormLayer);
#ifndef USE_CUDNN_BATCH_NORM
	REGISTER_LAYER_CLASS(CuDNNBatchNorm);
#endif

}


#endif