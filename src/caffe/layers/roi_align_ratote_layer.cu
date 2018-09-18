
#include <cfloat>
#include "caffe/layers/roi_align_rotated_layer.hpp"
//#include "sm_20_atomic_functions.hpp"
//#include "sm_60_atomic_functions.hpp"

#define M_PI       3.14159265358979323846   // pi
//#define __CUDACC__


namespace roi_align_ratote_gpu {

	// forward

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
	__global__ void RoIAlignRotatedForward(
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
		T* top_data) {
		CUDA_1D_KERNEL_LOOP(index, nthreads) {
			// (n, c, ph, pw) is an element in the pooled output
			int pw = index % pooled_width;
			int ph = (index / pooled_width) % pooled_height;
			int c = (index / pooled_width / pooled_height) % channels;
			int n = index / pooled_width / pooled_height / channels;

			const T* offset_bottom_rois = bottom_rois + n * 6;
			int roi_batch_ind = offset_bottom_rois[0];

			// Do not round
			T roi_center_w = offset_bottom_rois[1] * spatial_scale;
			T roi_center_h = offset_bottom_rois[2] * spatial_scale;
			T roi_width = offset_bottom_rois[3] * spatial_scale;
			T roi_height = offset_bottom_rois[4] * spatial_scale;
			T theta = offset_bottom_rois[5] * M_PI / 180.0;

			// Force malformed ROIs to be 1x1
			roi_width = max(roi_width, (T)1.);
			roi_height = max(roi_height, (T)1.);
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

			// roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
			// Appropriate translation needs to be applied after.
			T roi_start_h = -roi_height / 2.0;
			T roi_start_w = -roi_width / 2.0;
			T cosTheta = cos(theta);
			T sinTheta = sin(theta);

			// We do average (integral) pooling inside a bin
			const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

			T output_val = 0.;
			for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
			{
				const T yy = roi_start_h + ph * bin_size_h +
					static_cast<T>(iy + .5f) * bin_size_h /
					static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
				for (int ix = 0; ix < roi_bin_grid_w; ix++) {
					const T xx = roi_start_w + pw * bin_size_w +
						static_cast<T>(ix + .5f) * bin_size_w /
						static_cast<T>(roi_bin_grid_w);

					// Rotate by theta around the center and translate
					T x = xx * cosTheta + yy * sinTheta + roi_center_w;
					T y = yy * cosTheta - xx * sinTheta + roi_center_h;

					T val = bilinear_interpolate(
						offset_bottom_data, height, width, y, x, index);
					output_val += val;
				}
			}
			output_val /= count;

			top_data[index] = output_val;
		}
	}

	// backward

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

		w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

		return;
	}

	template <typename T>
	__global__ void RoIAlignRotatedBackward(
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

			const T* offset_bottom_rois = bottom_rois + n * 6;
			int roi_batch_ind = offset_bottom_rois[0];

			// Do not round
			T roi_center_w = offset_bottom_rois[1] * spatial_scale;
			T roi_center_h = offset_bottom_rois[2] * spatial_scale;
			T roi_width = offset_bottom_rois[3] * spatial_scale;
			T roi_height = offset_bottom_rois[4] * spatial_scale;
			T theta = offset_bottom_rois[5] * M_PI / 180.0;

			// Force malformed ROIs to be 1x1
			roi_width = max(roi_width, (T)1.);
			roi_height = max(roi_height, (T)1.);
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

			// roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
			// Appropriate translation needs to be applied after.
			T roi_start_h = -roi_height / 2.0;
			T roi_start_w = -roi_width / 2.0;
			T cosTheta = cos(theta);
			T sinTheta = sin(theta);

			// We do average (integral) pooling inside a bin
			const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

			for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
			{
				const T yy = roi_start_h + ph * bin_size_h +
					static_cast<T>(iy + .5f) * bin_size_h /
					static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
				for (int ix = 0; ix < roi_bin_grid_w; ix++) {
					const T xx = roi_start_w + pw * bin_size_w +
						static_cast<T>(ix + .5f) * bin_size_w /
						static_cast<T>(roi_bin_grid_w);

					// Rotate by theta around the center and translate
					T x = xx * cosTheta + yy * sinTheta + roi_center_w;
					T y = yy * cosTheta - xx * sinTheta + roi_center_h;

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
						gpu_atomic_add<T>(
							static_cast<T>(g1), offset_bottom_diff + y_low * width + x_low);
						gpu_atomic_add<T>(
							static_cast<T>(g2), offset_bottom_diff + y_low * width + x_high);
						gpu_atomic_add<T>(
							static_cast<T>(g3), offset_bottom_diff + y_high * width + x_low);
						gpu_atomic_add<T>(
							static_cast<T>(g4), offset_bottom_diff + y_high * width + x_high);
						/*atomicAdd(
							static_cast<T>(g1), offset_bottom_diff + y_low * width + x_low);
						atomicAdd(
							static_cast<T>(g2), offset_bottom_diff + y_low * width + x_high);
						atomicAdd(
							static_cast<T>(g3), offset_bottom_diff + y_high * width + x_low);
						atomicAdd(
							static_cast<T>(g4), offset_bottom_diff + y_high * width + x_high);*/
					} // if
				} // ix
			} // iy
		} // CUDA_1D_KERNEL_LOOP
	} // RoIAlignRotatedBackward

} // namespace

namespace caffe {


	template <typename Dtype>
	void ROIAlignRatotePoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* bottom_rois = bottom[1]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		//int* argmax_data = max_idx_.mutable_gpu_data();
		int count = top[0]->count();
		const int nthreads = count;
		int batch_size = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
		roi_align_ratote_gpu::RoIAlignRotatedForward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS >>>(
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
			top_data);
	}

	template <typename Dtype>
	void ROIAlignRatotePoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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
		roi_align_ratote_gpu::RoIAlignRotatedBackward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS >>> (
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
			bottom_rois);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(ROIAlignRatotePoolingLayer);

}  // namespace caffe