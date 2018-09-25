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

		const Dtype *scale_data;
		const Dtype *bias_data;
		Dtype *global_mean;
		Dtype *global_var;
		Dtype *save_mean;
		Dtype *save_inv_var;

		if (this->phase_ == TRAIN && !use_global_stats_) {
			global_mean = this->blobs_[0]->mutable_gpu_data();
			global_var = this->blobs_[1]->mutable_gpu_data();
			save_mean = save_mean_.mutable_gpu_data();
			save_inv_var = save_inv_var_.mutable_gpu_data();
		}
		else {
			global_mean = this->blobs_[0]->mutable_gpu_data();
			global_var = this->blobs_[1]->mutable_gpu_data();
		}
		if (this->scale_bias_) {
			scale_data = this->blobs_[3]->gpu_data();
			bias_data = this->blobs_[4]->gpu_data();
		}
		else {
			scale_data = scale_ones_.gpu_data();
			bias_data = bias_zeros_.gpu_data();
		}

		if (this->phase_ == TRAIN ) {
			Dtype factor = 1. - this->moving_average_fraction_;
			if (use_global_stats_) {
				factor = 0;
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
		Dtype * bottom_diff = bottom[0]->mutable_gpu_diff();
		const Dtype *bottom_data = bottom[0]->gpu_data();
		double epsilon = this->eps_;
		const Dtype* save_mean;
		const Dtype* save_inv_var;
		const Dtype* scale_data;
		Dtype*  scale_diff;
		Dtype*  bias_diff;

		save_mean = save_mean_.gpu_data();
		save_inv_var = save_inv_var_.gpu_data();
		if (this->scale_bias_) {
			scale_data = this->blobs_[3]->gpu_data();
			scale_diff = this->blobs_[3]->mutable_gpu_diff();
			bias_diff = this->blobs_[4]->mutable_gpu_diff();
		}
		else {
			scale_data = scale_ones_.gpu_data();
			scale_diff = scale_ones_.mutable_gpu_diff();
			bias_diff = bias_zeros_.mutable_gpu_diff();
		}

		CUDNN_CHECK(cudnnBatchNormalizationBackward(handle_, mode_,
			cudnn::dataType<Dtype>::one, cudnn::dataType<Dtype>::zero,
			cudnn::dataType<Dtype>::one, cudnn::dataType<Dtype>::one,
			bottom_desc_, bottom_data, bottom_desc_, top_diff, bottom_desc_, bottom_diff,
			scale_bias_mean_var_desc_, scale_data, scale_diff, bias_diff,
			CUDNN_BN_MIN_EPSILON, save_mean, save_inv_var));
		
		/*CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream(0)));*/
	}

	INSTANTIATE_LAYER_GPU_FUNCS(CuDNNBatchNormLayer);

}

#endif