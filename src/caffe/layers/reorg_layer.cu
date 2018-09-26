#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/reorg_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	__global__ void reorg_kernel(int N, const Dtype *x, int w, int h, int c, int batch, int stride, int forward, Dtype *out)
	{
		int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
		if (i >= N) return;
		int in_index = i;
		int in_w = i % w;
		i = i / w;
		int in_h = i % h;
		i = i / h;
		int in_c = i % c;
		i = i / c;
		int b = i % batch;

		int out_c = c / (stride * stride);

		int c2 = in_c % out_c;
		int offset = in_c / out_c;
		int w2 = in_w * stride + offset % stride;
		int h2 = in_h * stride + offset / stride;
		//printf("%d\n", offset);
		int out_index = w2 + w * stride * (h2 + h * stride * (c2 + out_c * b));

		// printf("%d %d %d\n", w2, h2, c2);
		//printf("%d %d\n", in_index, out_index);
		//if(out_index >= N || out_index < 0) printf("bad bad bad \n");

		if (forward)
			out[out_index] = x[in_index];
		else
			out[in_index] = x[out_index];
		//if(forward) out[1] = x[1];
		//else out[0] = x[0];
	}

	template <typename Dtype>
	void reorg_ongpu(const Dtype *x, int w, int h, int c, int batch, int stride, int forward, Dtype *out)
	{
		int size = w * h * c * batch;
		reorg_kernel <<<CAFFE_GET_BLOCKS(size), CAFFE_CUDA_NUM_THREADS>>>(size, x, w, h, c, batch, stride, forward, out);
		CUDA_POST_KERNEL_CHECK;
	}

	template <typename Dtype>
	void ReorgLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int N = bottom[0]->num();
		const Dtype *bottom_data = bottom[0]->gpu_data();
		Dtype *top_data = top[0]->mutable_gpu_data();
		reorg_ongpu(bottom_data, out_w, out_h, out_c, N, stride_, 1, top_data);
	}

	template <typename Dtype>
	void ReorgLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int N = bottom[0]->num();
		Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
		const Dtype *top_diff = top[0]->gpu_diff();
		reorg_ongpu(top_diff, out_w, out_h, out_c, N, stride_, 0, bottom_diff);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(ReorgLayer);

}