#ifdef USE_CUDNN
#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/cudnn_batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

	template<typename Dtype>
	void CuDNNBatchNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		const Dtype *bottom_data = bottom[0]->gpu_data();
		Dtype *top_data = top[0]->mutable_gpu_data();

		const void *scale_data;
		const void *bias_data;
		void *global_mean;
		void *global_var;
		void *save_mean;
		void *save_inv_var;

		if (this->phase_ == TRAIN) {
			global_mean = this->blobs_[0]->mutable_gpu_data();
			global_var = this->blobs_[1]->mutable_gpu_data();
			save_mean = save_mean_->mutable_gpu_data();
			save_inv_var = save_inv_var_->mutable_gpu_data();
		}
		else {
			global_mean = this->blobs_[0]->gpu_data();
			global_var = this->blobs_[1]->gpu_data();
		}
		if (this->scale_bias_) {
			scale_data = this->blobs_[3]->gpu_data();
			bias_data = this->blobs_[4]->gpu_data();
		}
		else {
			scale_data = scale_ones_->gpu_data();
			bias_data = bias_zeros_->gpu_data();
		}

		if (this->phase_ == TRAIN) {
			double factor = 1. - this->moving_average_fraction_;
			if (this->iter() == 0) {
				factor = 1.0;
			}
			CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(handle_, mode_,
				cudnn::dataType<Dtype>::one, cudnn::dataType<Dtype>::zero,
				bottom_desc_, bottom_data, top_desc_, top_data,
				scale_bias_mean_var_desc_, scale_data, bias_data,
				factor, global_mean, global_var, CUDNN_BN_MIN_EPSILON, save_mean, save_inv_var));
		}
		else if (this->phase_ == TEST) {
			CUDNN_CHECK(cudnnBatchNormalizationForwardInference(handle_,
				CUDNN_BATCHNORM_SPATIAL,
				cudnn::dataType<Dtype>::one, cudnn::dataType<Dtype>::zero,
				bottom_desc_, bottom_data, top_desc_, top_data,
				scale_bias_mean_var_desc_, scale_data, bias_data,
				global_mean, global_var, CUDNN_BN_MIN_EPSILON));
		}
		else {
			LOG(FATAL) << "Unknown phase";
		}
		//CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));

	}


	template <typename Dtype>
	void CuDNNBatchNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		const Dtype *top_diff = top[0]->gpu_diff();
		Dtype * bottom_diff = bottom[0]->mutbale_gpu_diff();
		const Dtype *bottom_data = bottom[0]->gpu_data();
		double epsilon = this->eps_;
		const void* save_mean;
		const void* save_inv_var;
		const void* scale_data;
		void*  scale_diff;
		void*  bias_diff;

		save_mean = save_mean_.gpu_data();
		save_inv_var = save_inv_var_->gpu_data();
		if (this->scale_bias_) {
			scale_data = this->blobs_[3]->gpu_data();
			scale_diff = this->blobs_[3]->mutable_gpu_diff();
			bias_diff = this->blobs_[4]->mutable_gpu_diff();
		}
		else {
			scale_data = scale_ones_->gpu_data();
			scale_diff = scale_ones_->mutable_gpu_diff();
			bias_diff = bias_zeros_->mutable_gpu_diff();
		}

		CUDNN_CHECK(cudnnBatchNormalizationBackward(handle_, mode_,
			cudnn::dataType<Dtype>::one, cudnn::dataType<Dtype>::zero,
			cudnn::dataType<Dtype>::one, cudnn::dataType<Dtype>::one,
			bottom_desc_, bottom_data, bottom_desc_, top_diff, bottom_desc_, bottom_diff,
			scale_bias_mean_var_desc_, scale_data, scale_diff, bias_diff,
			CUDNN_BN_MIN_EPSILON, save_mean, save_inv_var));
		/*CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream(0)));*/
	}

}

#endif