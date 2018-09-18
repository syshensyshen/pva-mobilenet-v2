
#include <cfloat>
#include <vector>

#include "caffe/layers/resize_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	__global__ void ResizeForwardKernel(
		const int count,
		const int num_channels,
		const int input_height,
		const int input_width,
		const int output_height,
		const int output_width,
		const Dtype height_scale,
		const Dtype width_scale,
		const Dtype* bottom_data,
		Dtype* top_data) {
		CUDA_KERNEL_LOOP(index, count) {

			int indexTemp = index;
			const int w = indexTemp % output_width;
			indexTemp /= output_width;
			const int h = indexTemp % output_height;
			indexTemp /= output_height;
			const int c = indexTemp % num_channels;
			indexTemp /= num_channels;
			const int n = indexTemp;

			const int in_y = fminf(h / height_scale, input_height - 1);
			const int in_x = fminf(w / width_scale, input_width - 1);
			top_data[index] =
				bottom_data[((n * num_channels + c) * input_height + in_y) * input_width + in_x];
		}
	}

	template <typename Dtype>
	__global__ void ResizeBackwardKernel(
		const int count,
		const int num_channels,
		const int input_height,
		const int input_width,
		const int output_height,
		const int output_width,
		const Dtype height_scale,
		const Dtype width_scale,
		const Dtype* top_diff,
		Dtype* bottom_diff,
		Dtype normalization) {
		CUDA_KERNEL_LOOP(index, count) {
			int indexTemp = index;
			const int x = indexTemp % input_width;
			indexTemp /= input_width;
			const int y = indexTemp % input_height;
			indexTemp /= input_height;
			const int c = indexTemp % num_channels;
			indexTemp /= num_channels;
			const int n = indexTemp;

			const int out_y = fminf(y / height_scale, output_height - 1);
			const int out_x = fminf(x / width_scale, output_width - 1);
			const int out_index =
				((n * num_channels + c) * output_height + out_y) * output_width + out_x;
#if __CUDA_ARCH__ >= 350
			atomicAdd(bottom_diff + out_index, __ldg(top_diff + index));
#else
			atomicAdd(bottom_diff + out_index, *(top_diff + index));
#endif
			bottom_diff[out_index] /= normalization;
		}
	}

	template <typename Dtype>
	__global__ void ResizeNormalizationKernel(const Dtype* top_diff, Dtype *normalization) {
		int index = threadIdx.x + blockDim.x * blockIdx.x;
		index += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
		*normalization += top_diff[index] * top_diff[index];
	}

	template <typename Dtype>
	void ResizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *>& bottom, const vector<Blob<Dtype> *>& top) {
		int batch = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
		const Dtype *bottom_data = bottom[0]->gpu_data();
		Dtype *top_data = top[0]->mutable_gpu_data();
		int count = bottom[0]->count();
		const int output_height = height * scale_h_;
		const int output_width = width * scale_w_;
		caffe_gpu_memset(top[0]->count(), Dtype(0.0), top_data);
		ResizeForwardKernel<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
			<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >>> (
				count,
				channels,
				height,
				width,
				output_height,
				output_width,
				scale_h_,
				scale_w_,
				bottom_data,
				top_data);
	}

	template <typename Dtype>
	void ResizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype> *>& bottom) {
		int batch = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
		Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
		const Dtype *top_diff = top[0]->gpu_diff();
		int count = bottom[0]->count();
		const int output_height = height * scale_h_;
		const int output_width = width * scale_w_;
		Dtype normalization = 0.0;
		/*ResizeNormalizationKernel<Dtype>
			<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> > (top_diff, &normization);*/
		caffe_gpu_l2norm(top[0]->count(), top[0]->gpu_data(), &normalization);
		ResizeBackwardKernel<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
			<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >>> (
				count,
				channels,
				height,
				width,
				output_height,
				output_width,
				scale_h_,
				scale_w_,
				top_diff,
				bottom_diff,
				normalization);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(ResizeLayer);
}