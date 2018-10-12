#include "caffe/layers/cosine_loss_layer.hpp"

namespace caffe {

	template<typename Dtype>
	__global__ void channels_gpu_l2_norm(const int n, const int channels, const Dtype* bottom,
		Dtype *norm_data) {
		CUDA_KERNEL_LOOP(index, n) {
			caffe_gpu_l2norm(channels, bottom + index * channels, norm_data);
		}
	}

	template <typename Dtype>
	void CosineLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		int batch = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();

		const Dtype *bottom_data = bottom[0]->gpu_data();
		Dtype *norm_data = norm_.mutable_gpu_data();
		Dtype *inner_product_data = inner_product.mutable_gpu_data();
		Dtype loss = Dtype(0.0);
		const Dtype* label = bottom[1]->cpu_data();

		caffe_gpu_gemm(CblasNoTrans, CblasTrans, batch, batch, channels, Dtype(1.0),
			bottom_data, bottom_data, Dtype(0.0), inner_product_data);

		for (size_t i = 0; i < batch; i++) {
			for (size_t j = i; j < batch; j++) {
				//inner_product_data[i * batch + j] = caffe_cpu_dot(channels, bottom_data + i * channels, bottom_data + j * channels);
				inner_product_data[i * batch + j] / (norm_data[i] * norm_data[j]);
				if (label[i] == label[j]) {
					loss += (1 - inner_product_data[i * batch + j]);
				}
				else {
					loss += inner_product_data[i * batch + j];
				}
			}
		}

		top[0]->mutable_cpu_data()[0] = loss;

	}

	template <typename Dtype>
	void CosineLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	}


	INSTANTIATE_LAYER_GPU_FUNCS(CosineLossLayer);

}