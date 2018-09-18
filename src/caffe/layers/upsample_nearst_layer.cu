
#include <cfloat>
#include <vector>

#include "caffe/layers/upsample_nearst_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	__device__ int translate_gpu_idx(int index, int channels, int height, int width, int scale_factor) {
		int x, y, z, w;
		w = index % width;
		index = index / width;
		z = index % height;
		index = index / height;
		y = index % channels;
		index = index / channels;
		x = index;
		w = w / scale_factor;
		z = z / scale_factor;
		height /= scale_factor;
		width /= scale_factor;
		return (((x*channels + y)*height) + z)*width + w;
	}

	__device__ int translate_gpu_idx_inv(
		int index, int channels, int height, int width, int scale_factor, int off_x, int off_y) {
		int x, y, z, w;
		w = index % width;
		index = index / width;
		z = index % height;
		index = index / height;
		y = index % channels;
		index = index / channels;
		x = index;
		w = w*scale_factor + off_x;
		z = z*scale_factor + off_y;
		height *= scale_factor;
		width *= scale_factor;
		return (((x*channels + y)*height) + z)*width + w;
	}

	/*template <typename Dtype>
	__global__ void UpsampleNormalizationKernel(Dtype* top_diff, Dtype *normalization, int count) {
		int index = threadIdx.x + blockDim.x * blockIdx.x;
		index += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
		if (index >= count)
			return;
		*normalization += top_diff[index] * top_diff[index];
	}*/

	template <typename Dtype>
	__global__ void UpsampleNearstForwardKernel(const Dtype *bottom_data, Dtype *top_data, int count,
		int scale_factor, int channels, int height, int width) {
		int index = threadIdx.x + blockDim.x * blockIdx.x;
		index += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
		if (index >= count)
			return;
		int ipidx = translate_gpu_idx(index, channels, height, width, scale_factor);
		top_data[index] = bottom_data[ipidx];
	}

	template <typename Dtype>
	__global__ void UpsampleNearstBackwardKernel(Dtype *bottom_diff, const Dtype *top_diff,
		int count, int scale_factor, int channels, int height,
		int width, Dtype normalization) {
		int index = threadIdx.x + blockDim.x * blockIdx.x;
		index += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
		if (index >= count)
			return;
		for (int i = 0; i < scale_factor; i++) {
			for (int j = 0; j < scale_factor; j++) {
				int ipidx = translate_gpu_idx_inv(index, channels, height, width, scale_factor, i, j);
				bottom_diff[index] += top_diff[ipidx] / normalization;
			}
		}
	}	

	template <typename Dtype>
	void UpsampleNearstLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int batch = top[0]->num();
		int channels = top[0]->channels();
		int height = top[0]->height();
		int width = top[0]->width();
		const Dtype *bottom_data = bottom[0]->gpu_data();
		Dtype *top_data = top[0]->mutable_gpu_data();
		int count = bottom[0]->count();
		caffe_gpu_memset(top[0]->count(), Dtype(0.0), top_data);
		UpsampleNearstForwardKernel< Dtype>
		<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >>> (
			bottom_data,
			top_data,
			count,
			spatial_scale_,
			channels,
			height,
			width
			);
	}

	template <typename Dtype>
	void UpsampleNearstLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int batch = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
		Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
		const Dtype *top_diff = top[0]->gpu_diff();
		int count = bottom[0]->count();
		//caffe_gpu_memset(count, Dtype(0.0), bottom_diff);
		Dtype normalization = 0.0;
		caffe_gpu_l2norm(top[0]->count(), top[0]->gpu_diff(), &normalization);
		UpsampleNearstBackwardKernel<Dtype>
			<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >>> (
				bottom_diff,
				top_diff,
				count,
				spatial_scale_,
				channels,
				height,
				width,
				normalization
				);

	}

	INSTANTIATE_LAYER_GPU_FUNCS(UpsampleNearstLayer);

}