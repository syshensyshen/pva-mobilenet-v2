// ------------------------------------------------------------------
// Copyright (c) 2018 syshen
// ------------------------------------------------------------------

#ifndef CAFFE_SYSHEN_L2LOSS_HPP_
#define CAFFE_SYSHEN_L2LOSS_HPP_

namespace caffe {

template <typename Dtype>
class SyshenL2LossLayer : public LossLayer<Dtype> {
 public:
  explicit SyshenL2LossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SyshenL2Loss"; }

  virtual inline int ExactNumBottomBlobs() const { return 2; }

  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
};

}  // namespace caffe

#endif  // CAFFE_FAST_RCNN_LAYERS_HPP_
