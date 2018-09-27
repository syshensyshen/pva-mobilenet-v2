#include <algorithm>
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void BatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		BatchNormParameter param = this->layer_param_.batch_norm_param();
		moving_average_fraction_ = param.moving_average_fraction();
		use_global_stats_ = this->phase_ == TEST;
		if (param.has_use_global_stats())
			use_global_stats_ = param.use_global_stats();
		if (bottom[0]->num_axes() == 1)
			channels_ = 1;
		else
			channels_ = bottom[0]->shape(1);
		eps_ = param.eps();
		//if (eps_ > CUDNN_BN_MIN_EPSILON) {
		//	eps_ = CUDNN_BN_MIN_EPSILON;
		//}
		scale_bias_ = false;
		scale_bias_ = param.scale_bias(); // by default = false;
		if (param.has_scale_filler() || param.has_bias_filler()) { // implicit set
			scale_bias_ = true;
		}
		if (this->blobs_.size() > 0) {
			LOG(INFO) << "Skipping parameter initialization";
		}
		else {
			vector<int> shape;
			vector<int> shape_scale_factor;
			shape_scale_factor.push_back(1);
			if (scale_bias_) {
				this->blobs_.resize(5);
			}
			else {
				this->blobs_.resize(3);
			}
			//this->blobs_.resize(3);

			shape.push_back(channels_);
			this->blobs_[0].reset(new Blob<Dtype>(shape));
			this->blobs_[1].reset(new Blob<Dtype>(shape));
			this->blobs_[2].reset(new Blob<Dtype>(shape_scale_factor));
			for (int i = 0; i < 3; ++i) {
				caffe_set(this->blobs_[i]->count(), Dtype(0),
					this->blobs_[i]->mutable_cpu_data());
			}
			if (scale_bias_) {
				this->blobs_[3].reset(new Blob<Dtype>(shape));
				this->blobs_[4].reset(new Blob<Dtype>(shape));
				if (param.has_scale_filler()) {
					shared_ptr<Filler<Dtype> > scale_filler(GetFiller<Dtype>(
						this->layer_param_.batch_norm_param().scale_filler()));
					scale_filler->Fill(this->blobs_[3].get());
				}
				else {
					caffe_set(this->blobs_[3]->count(), Dtype(1.0),
						this->blobs_[3]->mutable_cpu_data());
				}
				if (param.has_bias_filler()) {
					shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
						this->layer_param_.batch_norm_param().bias_filler()));
					bias_filler->Fill(this->blobs_[4].get());
				}
				else {
					caffe_set(this->blobs_[4]->count(), Dtype(0.0),
						this->blobs_[4]->mutable_cpu_data());
				}
			}
		}
		// Mask statistics from optimization by setting local learning rates
		// for mean, variance, and the bias correction to zero.
		for (int i = 0; i < 3; ++i) {
			if (this->layer_param_.param_size() == i) {
				ParamSpec* fixed_param_spec = this->layer_param_.add_param();
				fixed_param_spec->set_lr_mult(0.f);
			}
		}

		if (scale_bias_) {
			for (int i = 3; i < 5; ++i) {
				if (this->layer_param_.param_size() == i) {
					this->layer_param_.add_param();
				}
				//set lr and decay = 1 for scale and bias
				if (use_global_stats_) {
				    this->layer_param_.mutable_param(i)->set_lr_mult(0.0f);
				    this->layer_param_.mutable_param(i)->set_decay_mult(0.0f);
				} else {
					this->layer_param_.mutable_param(i)->set_lr_mult(1.f);
				    this->layer_param_.mutable_param(i)->set_decay_mult(1.f);
				}
			}
		}
		
		//LOG(INFO) << "########################################### cuda batch ";
		//LOG(INFO) << "########################################### " << bottom[0]->num();
		//LOG(INFO) << "########################################### " << bottom[0]->channels();
		//LOG(INFO) << "########################################### " << bottom[0]->height();
		//LOG(INFO) << "########################################### " << bottom[0]->width();
		
	}

	template <typename Dtype>
	void BatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		if (bottom[0]->num_axes() > 1)
			CHECK_EQ(bottom[0]->shape(1), channels_);
		top[0]->ReshapeLike(*bottom[0]);

		int N = bottom[0]->num();
		int C = bottom[0]->channels();
		int H = bottom[0]->height();
		int W = bottom[0]->width();
		vector<int> shape_c;
		shape_c.push_back(C);

		mean_.Reshape(shape_c);
		var_.Reshape(shape_c);
		inv_var_.Reshape(shape_c);
		temp_C_.Reshape(shape_c);
		vector<int> shape_n;
		shape_n.push_back(N);
		ones_N_.Reshape(shape_n);
		caffe_set(ones_N_.count(), Dtype(1.0),
			ones_N_.mutable_cpu_data());
		ones_C_.Reshape(shape_c);
		caffe_set(ones_C_.count(), Dtype(1.0),
			ones_C_.mutable_cpu_data());
		vector<int> shape_hw;
		shape_hw.push_back(H*W);
		ones_HW_.Reshape(shape_hw);
		caffe_set(ones_HW_.count(), Dtype(1.0),
			ones_HW_.mutable_cpu_data());
		vector<int> shape_nc;
		shape_nc.push_back(N*C);
		temp_NC_.Reshape(shape_nc);
		temp_NCHW_.ReshapeLike(*bottom[0]);
		x_norm_.ReshapeLike(*bottom[0]);
	}

	template <typename Dtype>
	void BatchNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int N = bottom[0]->num();
		int C = channels_;
		int HW = bottom[0]->count() / (N * C);
		int top_size = top[0]->count();

		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		const Dtype* global_mean = this->blobs_[0]->cpu_data();
		const Dtype* global_var = this->blobs_[1]->cpu_data();

		if (this->phase_ == TEST) {
			if (bottom[0] != top[0]) {
				caffe_copy(top_size, bottom_data, top_data);
			}
			//  Y = X- EX
			multicast_cpu(N, C, HW, global_mean, temp_NCHW_.mutable_cpu_data());
			caffe_axpy<Dtype>(top_size, Dtype(-1.), temp_NCHW_.mutable_cpu_data(),
				top_data);
			//  inv_var = (eps + var)^(-0.5)
			caffe_copy<Dtype>(C, global_var, var_.mutable_cpu_data());
			caffe_add_scalar<Dtype>(C, Dtype(eps_), var_.mutable_cpu_data());
			caffe_powx<Dtype>(C, var_.cpu_data(), Dtype(-0.5),
				inv_var_.mutable_cpu_data());
			//  X_norm = (X-EX) * inv_var
			multicast_cpu(N, C, HW, inv_var_.cpu_data(),
				temp_NCHW_.mutable_cpu_data());
			caffe_mul<Dtype>(top_size, top_data, temp_NCHW_.cpu_data(), top_data);
		}
		else {
			compute_mean_per_channel_cpu(N, C, HW, bottom_data,
				mean_.mutable_cpu_data());
			multicast_cpu(N, C, HW, mean_.mutable_cpu_data(),
				temp_NCHW_.mutable_cpu_data());
			//  Y = X- EX
			if (bottom[0] != top[0]) {
				caffe_copy(top_size, bottom_data, top_data);
			}
			caffe_axpy<Dtype>(top_size, Dtype(-1.), temp_NCHW_.mutable_cpu_data(),
				top_data);
			// compute variance E (X-EX)^2
			caffe_powx<Dtype>(top_size, top_data, Dtype(2.),
				temp_NCHW_.mutable_cpu_data());
			compute_mean_per_channel_cpu(N, C, HW, temp_NCHW_.mutable_cpu_data(),
				var_.mutable_cpu_data());
			//  inv_var= ( eps+ variance)^(-0.5)
			caffe_add_scalar<Dtype>(C, Dtype(eps_), var_.mutable_cpu_data());
			caffe_powx<Dtype>(C, var_.cpu_data(), Dtype(-0.5),
				inv_var_.mutable_cpu_data());
			// X_norm = (X-EX) * inv_var
			multicast_cpu(N, C, HW, inv_var_.cpu_data(),
				temp_NCHW_.mutable_cpu_data());
			caffe_mul<Dtype>(top_size, top_data, temp_NCHW_.cpu_data(), top_data);
			// copy top to x_norm for backward
			caffe_copy<Dtype>(top_size, top_data, x_norm_.mutable_cpu_data());

			// clip variance
			//  update global mean and variance
			if (iter_ > 1) {
				if (use_global_stats_) {
					moving_average_fraction_ = Dtype(0.0);
				}
				caffe_cpu_axpby<Dtype>(C, 1. - moving_average_fraction_,
					mean_.cpu_data(), moving_average_fraction_,
					this->blobs_[0]->mutable_cpu_data());
				caffe_cpu_axpby<Dtype>(C, 1. - moving_average_fraction_,
					var_.cpu_data(), moving_average_fraction_,
					this->blobs_[1]->mutable_cpu_data());
			}
			else {
				caffe_copy<Dtype>(C, mean_.cpu_data(),
					this->blobs_[0]->mutable_cpu_data());
				caffe_copy<Dtype>(C, var_.cpu_data(),
					this->blobs_[1]->mutable_cpu_data());
			}
			iter_++;
		}

		// -- STAGE 2:  Y = X_norm * scale[c] + shift[c]  -----------------
		if (scale_bias_) {
			// Y = X_norm * scale[c]
			//const Blob& scale_data = *(this->blobs_[3]);
			multicast_cpu(N, C, HW, this->blobs_[3]->cpu_data(),
				temp_NCHW_.mutable_cpu_data());
			caffe_mul<Dtype>(top_size, top_data, temp_NCHW_.cpu_data(), top_data);
			// Y = Y + shift[c]
			//const Blob& shift_data = *(this->blobs_[4]);
			multicast_cpu(N, C, HW, this->blobs_[4]->cpu_data(),
				temp_NCHW_.mutable_cpu_data());
			caffe_add<Dtype>(top_size, top_data, temp_NCHW_.mutable_cpu_data(), top_data);
		}
	}

	template <typename Dtype>
	void BatchNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		int N = bottom[0]->num();
		int C = channels_;
		int HW = bottom[0]->count() / (N * C);
		int top_size = top[0]->count();
		const Dtype* top_diff = top[0]->cpu_diff();

		// --  STAGE 1: compute dE/d(scale) and dE/d(shift) ---------------
		if (scale_bias_) {
			// scale_diff: dE/d(scale)  =  sum(dE/dY .* X_norm)
			Dtype* scale_diff = this->blobs_[3]->mutable_cpu_diff();
			caffe_mul<Dtype>(top_size, top_diff, x_norm_.cpu_data(),
				temp_NCHW_.mutable_cpu_diff());
			compute_sum_per_channel_cpu(N, C, HW, temp_NCHW_.cpu_diff(), scale_diff);
			// shift_diff: dE/d(shift) = sum (dE/dY)
			Dtype* shift_diff = this->blobs_[4]->mutable_cpu_diff();
			compute_sum_per_channel_cpu(N, C, HW, top_diff, shift_diff);

			// --  STAGE 2: backprop dE/d(x_norm) = dE/dY .* scale ------------
			// dE/d(X_norm) = dE/dY * scale[c]
			const Dtype* scale_data = this->blobs_[3]->cpu_data();
			multicast_cpu(N, C, HW, scale_data, temp_NCHW_.mutable_cpu_data());
			caffe_mul<Dtype>(top_size, top_diff, temp_NCHW_.cpu_data(),
				x_norm_.mutable_cpu_diff());
			top_diff = x_norm_.cpu_diff();
		}

		// --  STAGE 3: backprop dE/dY --> dE/dX --------------------------
		// ATTENTION: from now on we will use notation Y:= X_norm
		//
		// if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
		//    dE(Y)/dX =  (dE/dY - mean(dE/dY) - mean(dE/dY .* Y) .* Y) ./ sqrt(var(X) + eps)
		// where .* and ./ are element-wise product and division,
		//    mean, var, sum are computed along all dimensions except the channels.

		const Dtype* top_data = x_norm_.cpu_data();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		// temp = mean(dE/dY .* Y)
		caffe_mul<Dtype>(top_size, top_diff, top_data, temp_NCHW_.mutable_cpu_diff());
		compute_mean_per_channel_cpu(N, C, HW, temp_NCHW_.cpu_diff(),
			temp_C_.mutable_cpu_diff());
		multicast_cpu(N, C, HW, temp_C_.cpu_diff(),
			temp_NCHW_.mutable_cpu_diff());
		// bottom = mean(dE/dY .* Y) .* Y
		caffe_mul(top_size, temp_NCHW_.cpu_diff(), top_data, bottom_diff);
		// temp = mean(dE/dY)
		compute_mean_per_channel_cpu(N, C, HW, top_diff,
			temp_C_.mutable_cpu_diff());
		multicast_cpu(N, C, HW, temp_C_.cpu_diff(),
			temp_NCHW_.mutable_cpu_diff());
		// bottom = mean(dE/dY) + mean(dE/dY .* Y) .* Y
		caffe_add(top_size, temp_NCHW_.cpu_diff(), bottom_diff, bottom_diff);
		// bottom = dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
		caffe_cpu_axpby(top_size, Dtype(1.), top_diff, Dtype(-1.), bottom_diff);
		// dE/dX = dE/dX ./ sqrt(var(X) + eps)
		multicast_cpu(N, C, HW, inv_var_.cpu_data(),
			temp_NCHW_.mutable_cpu_data());
		caffe_mul(top_size, bottom_diff, temp_NCHW_.cpu_data(), bottom_diff);
	}


#ifdef CPU_ONLY
	STUB_GPU(BatchNormLayer);
#endif

	INSTANTIATE_CLASS(BatchNormLayer);
#ifndef USE_CUDNN_BATCH_NORM
	REGISTER_LAYER_CLASS(BatchNorm);
#endif
}  // namespace caffe
