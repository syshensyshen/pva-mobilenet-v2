#include <cfloat>
#include <algorithm>
#include <vector>

#include "caffe/layers/roi_align_layer.hpp"
using std::max;
using std::min;

namespace roi_align_gpu {

	template <typename T>
	__device__ T bilinear_interpolate(
		const T* bottom_data,
		const int height,
		const int width,
		T y,
		T x,
		const int index /* index for debug only*/) {
		// deal with cases that inverse elements are out of feature map boundary
		if (y < -1.0 || y > height || x < -1.0 || x > width) {
			// empty
			return 0;
		}

		if (y <= 0) {
			y = 0;
		}
		if (x <= 0) {
			x = 0;
		}

		int y_low = (int)y;
		int x_low = (int)x;
		int y_high;
		int x_high;

		if (y_low >= height - 1) {
			y_high = y_low = height - 1;
			y = (T)y_low;
		}
		else {
			y_high = y_low + 1;
		}

		if (x_low >= width - 1) {
			x_high = x_low = width - 1;
			x = (T)x_low;
		}
		else {
			x_high = x_low + 1;
		}

		T ly = y - y_low;
		T lx = x - x_low;
		T hy = 1. - ly, hx = 1. - lx;
		// do bilinear interpolation
		T v1 = bottom_data[y_low * width + x_low];
		T v2 = bottom_data[y_low * width + x_high];
		T v3 = bottom_data[y_high * width + x_low];
		T v4 = bottom_data[y_high * width + x_high];
		T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

		T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

		return val;
	}

	template <typename T>
	__global__ void RoIAlignForward(
		const int nthreads,
		const T* bottom_data,
		const T spatial_scale,
		const int channels,
		const int height,
		const int width,
		const int pooled_height,
		const int pooled_width,
		const int sampling_ratio,
		const T* bottom_rois,
		int roi_cols,
		T* top_data) {
		CUDA_1D_KERNEL_LOOP(index, nthreads) {
			// (n, c, ph, pw) is an element in the pooled output
			int pw = index % pooled_width;
			int ph = (index / pooled_width) % pooled_height;
			int c = (index / pooled_width / pooled_height) % channels;
			int n = index / pooled_width / pooled_height / channels;

			// RoI could have 4 or 5 columns
			const T* offset_bottom_rois = bottom_rois + n * roi_cols;
			int roi_batch_ind = 0;
			if (roi_cols == 5) {
				roi_batch_ind = offset_bottom_rois[0];
				offset_bottom_rois++;
			}

			// Do not using rounding; this implementation detail is critical
			T roi_start_w = offset_bottom_rois[0] * spatial_scale;
			T roi_start_h = offset_bottom_rois[1] * spatial_scale;
			T roi_end_w = offset_bottom_rois[2] * spatial_scale;
			T roi_end_h = offset_bottom_rois[3] * spatial_scale;
			// T roi_start_w = roundf(offset_bottom_rois[0] * spatial_scale);
			// T roi_start_h = roundf(offset_bottom_rois[1] * spatial_scale);
			// T roi_end_w = roundf(offset_bottom_rois[2] * spatial_scale);
			// T roi_end_h = roundf(offset_bottom_rois[3] * spatial_scale);

			// Force malformed ROIs to be 1x1
			T roi_width = max(roi_end_w - roi_start_w, (T)1.);
			T roi_height = max(roi_end_h - roi_start_h, (T)1.);
			T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
			T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

			const T* offset_bottom_data =
				bottom_data + (roi_batch_ind * channels + c) * height * width;

			// We use roi_bin_grid to sample the grid and mimic integral
			int roi_bin_grid_h = (sampling_ratio > 0)
				? sampling_ratio
				: ceil(roi_height / pooled_height); // e.g., = 2
			int roi_bin_grid_w =
				(sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

			// We do average (integral) pooling inside a bin
			const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

			T output_val = 0.;
			for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
			{
				const T y = roi_start_h + ph * bin_size_h +
					static_cast<T>(iy + .5f) * bin_size_h /
					static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
				for (int ix = 0; ix < roi_bin_grid_w; ix++) {
					const T x = roi_start_w + pw * bin_size_w +
						static_cast<T>(ix + .5f) * bin_size_w /
						static_cast<T>(roi_bin_grid_w);

					T val = bilinear_interpolate(
						offset_bottom_data, height, width, y, x, index);
					output_val += val;
				}
			}
			output_val /= count;

			top_data[index] = output_val;
		}
	}

	//backward
	template <typename T>
	inline __device__ T gpu_atomic_add(const T val, T* address);

	template <>
	inline __device__ float gpu_atomic_add(const float val, float* address) {
		return atomicAdd(address, val);
	}
	template <>
	inline __device__ double gpu_atomic_add(const double val, double* address) {
		return atomicAdd(address, val);
	}

	template <typename T>
	__device__ void bilinear_interpolate_gradient(
		const int height,
		const int width,
		T y,
		T x,
		T& w1,
		T& w2,
		T& w3,
		T& w4,
		int& x_low,
		int& x_high,
		int& y_low,
		int& y_high,
		const int index /* index for debug only*/) {
		// deal with cases that inverse elements are out of feature map boundary
		if (y < -1.0 || y > height || x < -1.0 || x > width) {
			// empty
			w1 = w2 = w3 = w4 = 0.;
			x_low = x_high = y_low = y_high = -1;
			return;
		}

		if (y <= 0) {
			y = 0;
		}
		if (x <= 0) {
			x = 0;
		}

		y_low = (int)y;
		x_low = (int)x;

		if (y_low >= height - 1) {
			y_high = y_low = height - 1;
			y = (T)y_low;
		}
		else {
			y_high = y_low + 1;
		}

		if (x_low >= width - 1) {
			x_high = x_low = width - 1;
			x = (T)x_low;
		}
		else {
			x_high = x_low + 1;
		}

		T ly = y - y_low;
		T lx = x - x_low;
		T hy = 1. - ly, hx = 1. - lx;

		// reference in forward
		// T v1 = bottom_data[y_low * width + x_low];
		// T v2 = bottom_data[y_low * width + x_high];
		// T v3 = bottom_data[y_high * width + x_low];
		// T v4 = bottom_data[y_high * width + x_high];
		// T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

		w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

		return;
	}

	template <typename T>
	__global__ void RoIAlignBackwardFeature(
		const int nthreads,
		const T* top_diff,
		const int num_rois,
		const T spatial_scale,
		const int channels,
		const int height,
		const int width,
		const int pooled_height,
		const int pooled_width,
		const int sampling_ratio,
		T* bottom_diff,
		const T* bottom_rois) {
		CUDA_1D_KERNEL_LOOP(index, nthreads) {
			// (n, c, ph, pw) is an element in the pooled output
			int pw = index % pooled_width;
			int ph = (index / pooled_width) % pooled_height;
			int c = (index / pooled_width / pooled_height) % channels;
			int n = index / pooled_width / pooled_height / channels;

			const T* offset_bottom_rois = bottom_rois + n * 5;
			int roi_batch_ind = offset_bottom_rois[0];

			// Do not using rounding; this implementation detail is critical
			T roi_start_w = offset_bottom_rois[1] * spatial_scale;
			T roi_start_h = offset_bottom_rois[2] * spatial_scale;
			T roi_end_w = offset_bottom_rois[3] * spatial_scale;
			T roi_end_h = offset_bottom_rois[4] * spatial_scale;
			// T roi_start_w = roundf(offset_bottom_rois[1] * spatial_scale);
			// T roi_start_h = roundf(offset_bottom_rois[2] * spatial_scale);
			// T roi_end_w = roundf(offset_bottom_rois[3] * spatial_scale);
			// T roi_end_h = roundf(offset_bottom_rois[4] * spatial_scale);

			// Force malformed ROIs to be 1x1
			T roi_width = max(roi_end_w - roi_start_w, (T)1.);
			T roi_height = max(roi_end_h - roi_start_h, (T)1.);
			T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
			T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

			T* offset_bottom_diff =
				bottom_diff + (roi_batch_ind * channels + c) * height * width;

			int top_offset = (n * channels + c) * pooled_height * pooled_width;
			const T* offset_top_diff = top_diff + top_offset;
			const T top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

			// We use roi_bin_grid to sample the grid and mimic integral
			int roi_bin_grid_h = (sampling_ratio > 0)
				? sampling_ratio
				: ceil(roi_height / pooled_height); // e.g., = 2
			int roi_bin_grid_w =
				(sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

			// We do average (integral) pooling inside a bin
			const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

			for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
			{
				const T y = roi_start_h + ph * bin_size_h +
					static_cast<T>(iy + .5f) * bin_size_h /
					static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
				for (int ix = 0; ix < roi_bin_grid_w; ix++) {
					const T x = roi_start_w + pw * bin_size_w +
						static_cast<T>(ix + .5f) * bin_size_w /
						static_cast<T>(roi_bin_grid_w);

					T w1, w2, w3, w4;
					int x_low, x_high, y_low, y_high;

					bilinear_interpolate_gradient(
						height,
						width,
						y,
						x,
						w1,
						w2,
						w3,
						w4,
						x_low,
						x_high,
						y_low,
						y_high,
						index);

					T g1 = top_diff_this_bin * w1 / count;
					T g2 = top_diff_this_bin * w2 / count;
					T g3 = top_diff_this_bin * w3 / count;
					T g4 = top_diff_this_bin * w4 / count;

					if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
						gpu_atomic_add(
							static_cast<T>(g1), offset_bottom_diff + y_low * width + x_low);
						gpu_atomic_add(
							static_cast<T>(g2), offset_bottom_diff + y_low * width + x_high);
						gpu_atomic_add(
							static_cast<T>(g3), offset_bottom_diff + y_high * width + x_low);
						gpu_atomic_add(
							static_cast<T>(g4), offset_bottom_diff + y_high * width + x_high);
					} // if
				} // ix
			} // iy
		} // CUDA_1D_KERNEL_LOOP
	} // RoIAlignBackward

}

namespace caffe {
	
template <typename Dtype>
void ROIAlignLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  const int nthreads = count;
  int batch_size = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int num_rois = bottom[1]->num();
  roi_align_gpu::RoIAlignForward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS >>> (
	  nthreads,
	  bottom_data,
	  spatial_scale_,
	  channels,
	  height,
	  width,
	  pooled_height_,
	  pooled_width_,
	  sampling_ratio_,
	  bottom_rois,
	  num_rois,
	  top_data);
}



template <typename Dtype>
void ROIAlignLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);

  //const int* argmax_data = max_idx_.gpu_data();
  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  const int nthreads = count;

  roi_align_gpu::RoIAlignBackwardFeature<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS >>> (
	  nthreads,
	  top_diff,
	  num_rois,
	  spatial_scale_,
	  channels,
	  height,
	  width,
	  pooled_height_,
	  pooled_width_,
	  sampling_ratio_,
	  bottom_diff,
	  bottom_rois
	  );
}

INSTANTIATE_LAYER_GPU_FUNCS(ROIAlignLayer);

}
