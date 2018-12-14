// ------------------------------------------------------------------
// Copyright (c) 2018 syshen
// ------------------------------------------------------------------

#include "caffe/layers/syshen_L1loss.hpp"

namespace caffe {

template <typename Dtype>
void SyshenL1LossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  SyshenL1LossParameter loss_param = this->layer_param_.syshen_l1_loss_param();
}

template <typename Dtype>
void SyshenL1LossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
 
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void SyshenL1LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void SyshenL1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(SyshenL1LossLayer);
#endif

INSTANTIATE_CLASS(SyshenL1LossLayer);
REGISTER_LAYER_CLASS(SyshenL1Loss);

}  // namespace caffe
