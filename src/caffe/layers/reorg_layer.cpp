#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/reorg_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void reorg_cpu(const Dtype *x, int out_w, int out_h, int out_c, int batch, int stride, int forward, Dtype *out)
	{
		int b, i, j, k;
		int in_c = out_c / (stride * stride);

		//printf("\n out_c = %d, out_w = %d, out_h = %d, stride = %d, forward = %d \n", out_c, out_w, out_h, stride, forward);
		//printf("  in_c = %d,  in_w = %d,  in_h = %d \n", in_c, out_w*stride, out_h*stride);

		for (b = 0; b < batch; ++b) {
			for (k = 0; k < out_c; ++k) {
				for (j = 0; j < out_h; ++j) {
					for (i = 0; i < out_w; ++i) {
						int in_index = i + out_w*(j + out_h*(k + out_c*b));
						int c2 = k % in_c;
						int offset = k / in_c;
						int w2 = i*stride + offset % stride;
						int h2 = j*stride + offset / stride;
						int out_index = w2 + out_w*stride*(h2 + out_h*stride*(c2 + in_c*b));
						if (forward) 
							out[out_index] = x[in_index];	// used by default for forward (i.e. forward = 0)
						else 
							out[in_index] = x[out_index];
					}
				}
			}
		}
	}

	template <typename Dtype>
	void ReorgLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const ReorgParameter & param = this->layer_param_.reorg_param();
		if (param.has_stride()) {
			stride_ = param.stride();
		}
		else {
			stride_ = Dtype(1.0f);
		}
	}

	template <typename Dtype>
	void ReorgLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		
		int N = bottom[0]->num();
		int C = bottom[0]->channels();
		int H = bottom[0]->height();
		int W = bottom[0]->width();

		out_h = H / stride_;
		out_w = W / stride_;
		out_c = C * stride_ * stride_;

		top[0]->Reshape(N, out_c, out_h, out_w);

	}

	template <typename Dtype>
	void ReorgLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int N = bottom[0]->num();
		const Dtype *bottom_data = bottom[0]->cpu_data();
		Dtype *top_data = top[0]->mutable_cpu_data();
		reorg_cpu(bottom_data, out_w, out_h, out_c, N, stride_, 1, top_data);
	}

	template <typename Dtype>
	void ReorgLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int N = bottom[0]->num();
		Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
		const Dtype *top_diff = top[0]->cpu_diff();
		reorg_cpu(top_diff, out_w, out_h, out_c, N, stride_, 0, bottom_diff);
	}

#ifdef CPU_ONLY
	STUB_GPU(CropLayer);
#endif

	INSTANTIATE_CLASS(ReorgLayer);
	REGISTER_LAYER_CLASS(Reorg);
}