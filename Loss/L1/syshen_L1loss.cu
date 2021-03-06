// ------------------------------------------------------------------
// Copyright (c) 2018 syshen
// ------------------------------------------------------------------

#include "caffe/layers/syshen_L1loss.hpp"

namespace caffe {

template <typename Dtype>
__global__ void L1Forward(const int n, const Dtype* in, Dtype* out) {
  // |x-label|
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = in[index];
    Dtype abs_val = abs(val);
    out[0] += abs_val;
  }
}

template <typename Dtype>
void SyshenL1LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());    // d := b0 - b1
  //Dtype *loss;
  //CUDA_CHECK(cudaMalloc((void **)&loss, sizeof(Dtype)));
  L1Forward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, diff_.gpu_data(), top[0]->mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK; 

  top[0]->mutable_cpu_data()[0] /= bottom[0]->num();
  //CUDA_CHECK(cudaFree(loss));
}

template <typename Dtype>
__global__ void L1Backward(const int n, const Dtype* in, Dtype* out) {
  // gradient = sign(x)                   otherwise
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = in[index];
    Dtype abs_val = abs(val);
    out[index] = (Dtype(0) < val) - (val < Dtype(0));
  }
}

template <typename Dtype>
void SyshenL1LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = diff_.count();
  L1Backward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, diff_.gpu_data(), diff_.mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
  for (int i = 0; i < 1; ++i) {
    if (propagate_down[i]) {
      CUDA_CHECK(cudaMemcpyAsync(bottom[0]->mutable_gpu_diff(), diff_.gpu_data(), 
       bottom[0]->count(), cudaMemcpyDeviceToDevice));
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SyshenL1LossLayer);

}  // namespace caffe
