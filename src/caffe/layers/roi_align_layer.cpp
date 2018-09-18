#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/roi_align_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace roi_align_cpu {

	//forweard
	template <typename T>
	struct PreCalc {
		int pos1;
		int pos2;
		int pos3;
		int pos4;
		T w1;
		T w2;
		T w3;
		T w4;
	};

	template <typename T>
	void pre_calc_for_bilinear_interpolate(
		const int height,
		const int width,
		const int pooled_height,
		const int pooled_width,
		const int iy_upper,
		const int ix_upper,
		T roi_start_h,
		T roi_start_w,
		T bin_size_h,
		T bin_size_w,
		int roi_bin_grid_h,
		int roi_bin_grid_w,
		std::vector<PreCalc<T>>& pre_calc) {
		int pre_calc_index = 0;
		for (int ph = 0; ph < pooled_height; ph++) {
			for (int pw = 0; pw < pooled_width; pw++) {
				for (int iy = 0; iy < iy_upper; iy++) {
					const T yy = roi_start_h + ph * bin_size_h +
						static_cast<T>(iy + .5f) * bin_size_h /
						static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
					for (int ix = 0; ix < ix_upper; ix++) {
						const T xx = roi_start_w + pw * bin_size_w +
							static_cast<T>(ix + .5f) * bin_size_w /
							static_cast<T>(roi_bin_grid_w);

						T x = xx;
						T y = yy;
						// deal with: inverse elements are out of feature map boundary
						if (y < -1.0 || y > height || x < -1.0 || x > width) {
							// empty
							PreCalc<T> pc;
							pc.pos1 = 0;
							pc.pos2 = 0;
							pc.pos3 = 0;
							pc.pos4 = 0;
							pc.w1 = 0;
							pc.w2 = 0;
							pc.w3 = 0;
							pc.w4 = 0;
							pre_calc[pre_calc_index] = pc;
							pre_calc_index += 1;
							continue;
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
						T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

						// save weights and indeces
						PreCalc<T> pc;
						pc.pos1 = y_low * width + x_low;
						pc.pos2 = y_low * width + x_high;
						pc.pos3 = y_high * width + x_low;
						pc.pos4 = y_high * width + x_high;
						pc.w1 = w1;
						pc.w2 = w2;
						pc.w3 = w3;
						pc.w4 = w4;
						pre_calc[pre_calc_index] = pc;

						pre_calc_index += 1;
					}
				}
			}
		}
	}

	template <typename T>
	void ROIAlignForward(
		const int nthreads,
		const T* bottom_data,
		const T& spatial_scale,
		const int channels,
		const int height,
		const int width,
		const int pooled_height,
		const int pooled_width,
		const int sampling_ratio,
		const T* bottom_rois,
		int roi_cols,
		T* top_data) {
		DCHECK(roi_cols == 4 || roi_cols == 5);

		int n_rois = nthreads / channels / pooled_width / pooled_height;

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int n = 0; n < n_rois; n++) {
			int index_n = n * channels * pooled_width * pooled_height;

			// roi could have 4 or 5 columns
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
			// T roi_start_w = round(offset_bottom_rois[0] * spatial_scale);
			// T roi_start_h = round(offset_bottom_rois[1] * spatial_scale);
			// T roi_end_w = round(offset_bottom_rois[2] * spatial_scale);
			// T roi_end_h = round(offset_bottom_rois[3] * spatial_scale);

			// Force malformed ROIs to be 1x1
			T roi_width = std::max(roi_end_w - roi_start_w, (T)1.);
			T roi_height = std::max(roi_end_h - roi_start_h, (T)1.);
			T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
			T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

			// We use roi_bin_grid to sample the grid and mimic integral
			int roi_bin_grid_h = (sampling_ratio > 0)
				? sampling_ratio
				: ceil(roi_height / pooled_height); // e.g., = 2
			int roi_bin_grid_w =
				(sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

			// We do average (integral) pooling inside a bin
			const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

															 // we want to precalculate indeces and weights shared by all chanels,
															 // this is the key point of optimiation
			std::vector<PreCalc<T>> pre_calc(
				roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
			pre_calc_for_bilinear_interpolate(
				height,
				width,
				pooled_height,
				pooled_width,
				roi_bin_grid_h,
				roi_bin_grid_w,
				roi_start_h,
				roi_start_w,
				bin_size_h,
				bin_size_w,
				roi_bin_grid_h,
				roi_bin_grid_w,
				pre_calc);

			for (int c = 0; c < channels; c++) {
				int index_n_c = index_n + c * pooled_width * pooled_height;
				const T* offset_bottom_data =
					bottom_data + (roi_batch_ind * channels + c) * height * width;
				int pre_calc_index = 0;

				for (int ph = 0; ph < pooled_height; ph++) {
					for (int pw = 0; pw < pooled_width; pw++) {
						int index = index_n_c + ph * pooled_width + pw;

						T output_val = 0.;
						for (int iy = 0; iy < roi_bin_grid_h; iy++) {
							for (int ix = 0; ix < roi_bin_grid_w; ix++) {
								PreCalc<T> pc = pre_calc[pre_calc_index];
								output_val += pc.w1 * offset_bottom_data[pc.pos1] +
									pc.w2 * offset_bottom_data[pc.pos2] +
									pc.w3 * offset_bottom_data[pc.pos3] +
									pc.w4 * offset_bottom_data[pc.pos4];

								pre_calc_index += 1;
							}
						}
						output_val /= count;

						top_data[index] = output_val;
					} // for pw
				} // for ph
			} // for c
		} // for n
	}

	//backward

	template <typename T>
	void bilinear_interpolate_gradient(
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
		const int /*index*/ /* index for debug only*/) {
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

	template <class T>
	inline void add(const T& val, T* address) {
		*address += val;
	}

	template <typename T>
	void ROIAlignBackwardFeature(
		const int nthreads,
		const T* top_diff,
		const int /*num_rois*/,
		const T& spatial_scale,
		const int channels,
		const int height,
		const int width,
		const int pooled_height,
		const int pooled_width,
		const int sampling_ratio,
		T* bottom_diff,
		const T* bottom_rois,
		int rois_cols) {
		DCHECK(rois_cols == 4 || rois_cols == 5);

		for (int index = 0; index < nthreads; index++) {
			// (n, c, ph, pw) is an element in the pooled output
			int pw = index % pooled_width;
			int ph = (index / pooled_width) % pooled_height;
			int c = (index / pooled_width / pooled_height) % channels;
			int n = index / pooled_width / pooled_height / channels;

			const T* offset_bottom_rois = bottom_rois + n * rois_cols;
			int roi_batch_ind = 0;
			if (rois_cols == 5) {
				roi_batch_ind = offset_bottom_rois[0];
				offset_bottom_rois++;
			}

			// Do not using rounding; this implementation detail is critical
			T roi_start_w = offset_bottom_rois[0] * spatial_scale;
			T roi_start_h = offset_bottom_rois[1] * spatial_scale;
			T roi_end_w = offset_bottom_rois[2] * spatial_scale;
			T roi_end_h = offset_bottom_rois[3] * spatial_scale;
			// T roi_start_w = round(offset_bottom_rois[0] * spatial_scale);
			// T roi_start_h = round(offset_bottom_rois[1] * spatial_scale);
			// T roi_end_w = round(offset_bottom_rois[2] * spatial_scale);
			// T roi_end_h = round(offset_bottom_rois[3] * spatial_scale);

			// Force malformed ROIs to be 1x1
			T roi_width = std::max(roi_end_w - roi_start_w, (T)1.);
			T roi_height = std::max(roi_end_h - roi_start_h, (T)1.);
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

			for (int iy = 0; iy < roi_bin_grid_h; iy++) {
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
						// atomic add is not needed for now since it is single threaded
						add(static_cast<T>(g1), offset_bottom_diff + y_low * width + x_low);
						add(static_cast<T>(g2), offset_bottom_diff + y_low * width + x_high);
						add(static_cast<T>(g3), offset_bottom_diff + y_high * width + x_low);
						add(static_cast<T>(g4), offset_bottom_diff + y_high * width + x_high);
					} // if
				} // ix
			} // iy
		} // for
	} // ROIAlignBackward
}

namespace caffe {

	template <typename Dtype>
	void ROIAlignLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		ROIAlignParameter roi_align_param = this->layer_param_.roi_align_param();
		CHECK_GT(roi_align_param.pooled_h(), 0)
			<< "pooled_h must be > 0";
		CHECK_GT(roi_align_param.pooled_w(), 0)
			<< "pooled_w must be > 0";
		pooled_height_ = roi_align_param.pooled_h();
		pooled_width_ = roi_align_param.pooled_w();
		spatial_scale_ = roi_align_param.spatial_scale();
		sampling_ratio_ = roi_align_param.sampling_ratio();
		LOG(INFO) << "Spatial scale: " << spatial_scale_;
	}

	template <typename Dtype>
	void ROIAlignLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		channels_ = bottom[0]->channels();
		height_ = bottom[0]->height();
		width_ = bottom[0]->width();
		top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,
			pooled_width_);
	}

	template <typename Dtype>
	void ROIAlignLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* bottom_rois = bottom[1]->cpu_data();
		// Number of ROIs
		int num_rois = bottom[1]->num();
		int batch_size = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
		int top_count = top[0]->count();
		Dtype* top_data = top[0]->mutable_cpu_data();
		caffe_set(top_count, Dtype(-FLT_MAX), top_data);
		//int* argmax_data = max_idx_.mutable_cpu_data();
		//caffe_set(top_count, -1, argmax_data);
		const int count = bottom[1]->count();
		const int nthreads = count;
		roi_align_cpu::ROIAlignForward<Dtype>(
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
			top_data
			);

	}

	template <typename Dtype>
	void ROIAlignLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* bottom_rois = bottom[1]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		const int count = bottom[0]->count();
		caffe_gpu_set(count, Dtype(0.), bottom_diff);

		//const int* argmax_data = max_idx_.gpu_data();
		int num_rois = bottom[1]->num();
		int batch_size = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
		const int nthreads = count;
		roi_align_cpu::ROIAlignBackwardFeature<Dtype>(
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
			bottom_rois,
			num_rois);
	}


#ifdef CPU_ONLY
	STUB_GPU(ROIAlignLayer);
#endif

	INSTANTIATE_CLASS(ROIAlignLayer);
	REGISTER_LAYER_CLASS(ROIAlign);

}  // namespace caffe
