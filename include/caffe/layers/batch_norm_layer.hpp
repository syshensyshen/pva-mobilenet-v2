#ifndef CAFFE_BATCHNORM_LAYER_HPP_
#define CAFFE_BATCHNORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Normalizes the input to have 0-mean and/or unit (1) variance across
 *        the batch.
 *
 * This layer computes Batch Normalization as described in [1]. For each channel
 * in the data (i.e. axis 1), it subtracts the mean and divides by the variance,
 * where both statistics are computed across both spatial dimensions and across
 * the different examples in the batch.
 *
 * By default, during training time, the network is computing global
 * mean/variance statistics via a running average, which is then used at test
 * time to allow deterministic outputs for each input. You can manually toggle
 * whether the network is accumulating or using the statistics via the
 * use_global_stats option. For reference, these statistics are kept in the
 * layer's three blobs: (0) mean, (1) variance, and (2) moving average factor.
 *
 * Note that the original paper also included a per-channel learned bias and
 * scaling factor. To implement this in Caffe, define a `ScaleLayer` configured
 * with `bias_term: true` after each `BatchNormLayer` to handle both the bias
 * and scaling factor.
 *
 * [1] S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network
 *     Training by Reducing Internal Covariate Shift." arXiv preprint
 *     arXiv:1502.03167 (2015).
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class BatchNormLayer : public Layer<Dtype> {
 public:
  explicit BatchNormLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BatchNorm"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // multicast x[c] into y[.,c,...]
  template <typename Dtype>
  void multicast_gpu(int N, int C, int S, const Dtype *x, Dtype *y) {
	  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N, C, 1, Dtype(1.),
		  ones_N_.gpu_data(), x, Dtype(0.),
		  temp_NC_.mutable_gpu_data());
	  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N*C, S, 1, Dtype(1.),
		  temp_NC_.gpu_data(), ones_HW_.gpu_data(), Dtype(0.), y);
  }

  // y[c] = sum x(.,c,...)
  template <typename Dtype>
  void compute_sum_per_channel_gpu(int N, int C, int S, const Dtype *x, Dtype *y) {
	  caffe_gpu_gemv<Dtype>(CblasNoTrans, N * C, S, Dtype(1.), x,
		  ones_HW_.gpu_data(),
		  Dtype(0.), temp_NC_.mutable_gpu_data());
	  caffe_gpu_gemv<Dtype>(CblasTrans, N, C, Dtype(1.), temp_NC_.gpu_data(),
		  ones_N_.gpu_data(), Dtype(0.), y);
  }

  // y[c] = mean x(.,c,...)
  template <typename Dtype>
  void compute_mean_per_channel_gpu(int N, int C, int S, const Dtype *x, Dtype *y) {
	  Dtype F = 1. / (N * S);
	  compute_sum_per_channel_gpu(N, C, S, x, y);
	  caffe_gpu_scal(C, F, y);
  }

  //  multicast x[c] into y[.,c,...]
  template <typename Dtype>
  void multicast_cpu(int N, int C, int S, const Dtype *x, Dtype *y) {
	  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N, C, 1, Dtype(1.),
		  ones_N_.cpu_data(), x, Dtype(0.),
		  temp_NC_.mutable_cpu_data());
	  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N*C, S, 1,
		  Dtype(1.), temp_NC_.cpu_data(), ones_HW_.cpu_data(),
		  Dtype(0.), y);
  }

  //  y[c] = sum x(.,c,...)
  template <typename Dtype>
  void compute_sum_per_channel_cpu(int N, int C, int S, const Dtype *x, Dtype *y) {
	  caffe_cpu_gemv<Dtype>(CblasNoTrans, N * C, S, Dtype(1.), x,
		  ones_HW_.cpu_data(), Dtype(0.),
		  temp_NC_.mutable_cpu_data());
	  caffe_cpu_gemv<Dtype>(CblasTrans, N, C, Dtype(1.), temp_NC_.cpu_data(),
		  ones_N_.cpu_data(), Dtype(0.), y);
  }

  // y[c] = mean x(.,c,...)
  template <typename Dtype>
  void compute_mean_per_channel_cpu(int N, int C, int S, const Dtype *x, Dtype *y) {
	  Dtype F = 1. / (N * S);
	  compute_sum_per_channel_cpu(N, C, S, x, y);
	  caffe_cpu_scale(C, F, y, y);
  }

  Blob<Dtype> mean_, variance_, temp_, x_norm_;
  bool use_global_stats_;
  Dtype moving_average_fraction_;
  int channels_;
  Dtype eps_;

  // extra temporarary variables is used to carry out sums/broadcasting
  // using BLAS
  Blob<Dtype> batch_sum_multiplier_;
  Blob<Dtype> num_by_chans_;
  Blob<Dtype> spatial_sum_multiplier_;

  int iter_;
  bool clip_variance_, scale_bias_;
  Blob<Dtype> var_, inv_var_;
  // auxiliary arrays used for sums and broadcast
  Blob<Dtype> ones_N_, ones_HW_, ones_C_, temp_C_, temp_NC_, temp_NCHW_;

};

}  // namespace caffe

#endif  // CAFFE_BATCHNORM_LAYER_HPP_
