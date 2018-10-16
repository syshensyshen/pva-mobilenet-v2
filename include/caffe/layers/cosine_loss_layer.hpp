#ifndef CAFFE_COSINE_LOSS_LAYER_HPP_
#define CAFFE_COSINE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"


namespace caffe {
	
	template <typename Dtype>
	class CosineLossLayer :public LossLayer<Dtype> {

	public:
		explicit CosineLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual inline const char* type() const { return "CosineLoss"; }
		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int MinTopBlobs() const { return 1; }

	protected:

		void accu_assign(int batch, int count, bool reverse, const Dtype *x, Dtype *y, Dtype normalization) {
			for (size_t i = 0; i < count; i++) {
				normalization = normalization < 0.01 ? 0.01 : normalization;
				Dtype x_norm = x[i] / normalization / (batch - 1);
				Dtype scale = reverse ? pos_weight : nos_weight;
				x_norm = x_norm * scale;
				//x_norm = x_norm > 1 ? x_norm : Dtype(1.0);
				y[i] += reverse ? -x_norm : x_norm;
			}
		}
	protected:
		/// @copydoc AbsValLayer
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		/*virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);*/

		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		/*virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);*/
	private:
		Blob<Dtype> norm_;
		Blob<Dtype> inner_product;
		Dtype pos_weight;
		Dtype nos_weight;
	};

	/*template <>
	void accu_assign(int count, bool reverse, float *x, float *y, float normalization);
	template <>
	void accu_assign(int count, bool reverse, double *x, double *y, double normalization);*/

}



#endif