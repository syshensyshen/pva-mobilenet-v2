#ifndef CAFFE_UPSAMPLE_NERAST_LAYER_HPP_
#define CAFFE_UPSAMPLE_NERAST_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
	template <typename Dtype> 
	class UpsampleNearstLayer : public Layer<Dtype> {
	public:
		explicit UpsampleNearstLayer(const LayerParameter  &param)
			: Layer<Dtype>(param){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	public:

		virtual inline const char* type() const { return "UpsampleNearst"; }
		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int MinTopBlobs() const { return 1; }
// 		virtual inline int ExactNumBottomBlobs() const { return 1; }
// 		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	protected:
		int spatial_scale_;

	};
}

#endif
