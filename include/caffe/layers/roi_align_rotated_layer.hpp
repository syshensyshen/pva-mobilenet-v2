#ifndef CAFFE_ROI_ALIGN_RATOTE_LAYERS_HPP_
#define CAFFE_ROI_ALIGN_RATOTE_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	/* ROIAlignRatoteLayer - Region of Interest Ratote Pooling Layer
	*/
	template <typename Dtype>
	class ROIAlignRatotePoolingLayer : public Layer<Dtype> {
	public:
		explicit ROIAlignRatotePoolingLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "ROIPooling"; }

		virtual inline int MinBottomBlobs() const { return 2; }
		virtual inline int MaxBottomBlobs() const { return 2; }
		virtual inline int MinTopBlobs() const { return 1; }
		virtual inline int MaxTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int channels_;
		int height_;
		int width_;
		int pooled_height_;
		int pooled_width_;
		Dtype spatial_scale_;
		Dtype sampling_ratio_;
		//Blob<int> max_idx_;
	};
}

#endif
