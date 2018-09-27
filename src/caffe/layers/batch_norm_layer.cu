#include <algorithm>
#include <vector>

#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {



	template <typename Dtype>
	void BatchNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int N = bottom[0]->shape(0);
		int C = channels_;
		int S = bottom[0]->count(0) / (N * C);
		int top_size = top[0]->count();

		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		const Dtype* global_mean = this->blobs_[0]->gpu_data();
		const Dtype* global_var = this->blobs_[1]->gpu_data();

		if (this->phase_ == TEST) {
			//  Y = X- EX
			multicast_gpu(N, C, S, global_mean, temp_NCHW_.mutable_gpu_data());
			caffe_gpu_sub<Dtype>(top_size, bottom_data, temp_NCHW_.gpu_data(), top_data);
			//  inv_var = (eps + var)^(-0.5)
			caffe_copy<Dtype>(C, global_var, var_.mutable_gpu_data());
			caffe_gpu_add_scalar<Dtype>(C, Dtype(eps_), var_.mutable_gpu_data());
			caffe_gpu_powx<Dtype>(C, var_.gpu_data(), Dtype(-0.5F),
				inv_var_.mutable_gpu_data());
			//  X_norm = (X-EX) * inv_var
			multicast_gpu(N, C, S, inv_var_.gpu_data(),
				temp_NCHW_.mutable_gpu_data());
			caffe_gpu_mul<Dtype>(top_size, top_data, temp_NCHW_.gpu_data(), top_data);
		}
		else {  
			// if (this->phase_ == TRAIN)
			// temp = EX
			compute_mean_per_channel_gpu(N, C, S, bottom_data,
				mean_.mutable_gpu_data());
			multicast_gpu(N, C, S, mean_.gpu_data(),
				temp_NCHW_.mutable_gpu_data());
			// Y = X-EX
			caffe_gpu_sub<Dtype>(top_size, bottom_data, temp_NCHW_.gpu_data(), top_data);
			// temp = (X-EX)^2;
			caffe_gpu_square<Dtype>(top_size, top[0]->gpu_data(),
				temp_NCHW_.mutable_gpu_data());
			compute_mean_per_channel_gpu(N, C, S, temp_NCHW_.gpu_data(),
				var_.mutable_gpu_data());

			caffe_copy<Dtype>(C, var_.gpu_data(),
				temp_C_.mutable_gpu_data());
			//  temp= 1/sqrt(e + var(c)
			caffe_gpu_add_scalar<Dtype>(C, Dtype(eps_), temp_C_.mutable_gpu_data());
			caffe_gpu_powx<Dtype>(C, temp_C_.gpu_data(), Dtype(-0.5F),
				inv_var_.mutable_gpu_data());
			multicast_gpu(N, C, S, inv_var_.gpu_data(),
				temp_NCHW_.mutable_gpu_data());
			// X_norm = (X-mean(c)) / sqrt(e + var(c))
			caffe_gpu_mul<Dtype>(top_size, top_data, temp_NCHW_.gpu_data(), top_data);
			// copy x_norm for backward
			caffe_copy<Dtype>(top_size, top_data, x_norm_.mutable_gpu_data());

			//  update global mean and variance
			if (iter_ > 1) {
				if (use_global_stats_) {
					moving_average_fraction_ = Dtype(0.0);
				}
				caffe_gpu_axpby<Dtype>(C, 1. - moving_average_fraction_,
					mean_.gpu_data(), moving_average_fraction_,
					this->blobs_[0]->mutable_gpu_data());
				caffe_gpu_axpby<Dtype>(C, 1. - moving_average_fraction_,
					var_.gpu_data(), moving_average_fraction_,
					this->blobs_[1]->mutable_gpu_data());
			}
			else {
				caffe_copy<Dtype>(C, mean_.gpu_data(),
					this->blobs_[0]->mutable_gpu_data());
				caffe_copy<Dtype>(C, var_.gpu_data(),
					this->blobs_[1]->mutable_gpu_data());
			}
			iter_++;
		}

		//  -- STAGE 2:  Y = X_norm * scale[c] + shift[c]  -----------------
		if (scale_bias_) {
			//  Y = X_norm * scale[c]
			multicast_gpu(N, C, S, this->blobs_[3]->gpu_data(),
				temp_NCHW_.mutable_gpu_data());
			caffe_gpu_mul<Dtype>(top_size, top_data, temp_NCHW_.gpu_data(), top_data);
			//  Y = Y + shift[c]
			multicast_gpu(N, C, S, this->blobs_[4]->gpu_data(),
				temp_NCHW_.mutable_gpu_data());
			caffe_gpu_add<Dtype>(top_size, top_data, temp_NCHW_.mutable_gpu_data(),
				top_data);
		}
	}

	template <typename Dtype>
	void BatchNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		int N = bottom[0]->num();
		int C = channels_;
		int HW = bottom[0]->count(0) / (N * C);
		int top_size = top[0]->count();

		const Dtype* top_diff = top[0]->gpu_diff();
		//  --  STAGE 1: compute dE/d(scale) and dE/d(shift) ---------------
		if (scale_bias_) {
			//  scale_diff: dE/d(scale)  =  sum(dE/dY .* X_norm)
			Dtype* scale_diff = this->blobs_[3]->mutable_gpu_diff();
			caffe_gpu_mul<Dtype>(top_size, top_diff, x_norm_.gpu_data(),
				temp_NCHW_.mutable_gpu_diff());
			compute_sum_per_channel_gpu(N, C, HW, temp_NCHW_.gpu_diff(), scale_diff);
			//  shift_diff: dE/d(shift) = sum (dE/dY)
			Dtype* shift_diff = this->blobs_[4]->mutable_gpu_diff();
			compute_sum_per_channel_gpu(N, C, HW, top_diff, shift_diff);

			// --  STAGE 2: backprop dE/d(x_norm) = dE/dY .* scale ------------
			//  dE/d(X_norm) = dE/dY * scale[c]
			const Dtype* scale_data = this->blobs_[3]->gpu_data();
			multicast_gpu(N, C, HW, scale_data, temp_NCHW_.mutable_gpu_data());
			caffe_gpu_mul<Dtype>(top_size, top_diff, temp_NCHW_.gpu_data(),
				x_norm_.mutable_gpu_diff());

			top_diff = x_norm_.gpu_diff();
		}
		// --  STAGE 3: backprop dE/dY --> dE/dX --------------------------

		// ATTENTION: from now on we will use notation Y:= X_norm
		const Dtype* top_data = x_norm_.gpu_data();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

		//  temp = mean(dE/dY .* Y)
		caffe_gpu_mul<Dtype>(top_size, top_diff, top_data,
			temp_NCHW_.mutable_gpu_diff());
		compute_mean_per_channel_gpu(N, C, HW, temp_NCHW_.gpu_diff(),
			temp_C_.mutable_gpu_diff());
		multicast_gpu(N, C, HW, temp_C_.gpu_diff(),
			temp_NCHW_.mutable_gpu_diff());

		// bottom = mean(dE/dY .* Y) .* Y
		caffe_gpu_mul<Dtype>(top_size, temp_NCHW_.gpu_diff(), top_data, bottom_diff);

		// temp = mean(dE/dY)
		compute_mean_per_channel_gpu(N, C, HW, top_diff,
			temp_C_.mutable_gpu_diff());
		multicast_gpu(N, C, HW, temp_C_.gpu_diff(),
			temp_NCHW_.mutable_gpu_diff());

		// bottom = mean(dE/dY) + mean(dE/dY .* Y) .* Y
		caffe_gpu_add<Dtype>(top_size, temp_NCHW_.gpu_diff(), bottom_diff, bottom_diff);

		// bottom = dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
		caffe_gpu_sub<Dtype>(top_size, top_diff, bottom_diff, bottom_diff);

		// dE/dX = dE/dX ./ sqrt(var(X) + eps)
		multicast_gpu(N, C, HW, inv_var_.gpu_data(),
			temp_NCHW_.mutable_gpu_data());
		caffe_gpu_mul<Dtype>(top_size, bottom_diff, temp_NCHW_.gpu_data(), bottom_diff);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(BatchNormLayer);

}  // namespace caffe
