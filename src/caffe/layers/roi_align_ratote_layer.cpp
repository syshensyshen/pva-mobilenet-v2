#include <cfloat>

#include "caffe/layers/roi_align_rotated_layer.hpp"

#ifdef _MSC_VER
#define round(x) ((int)((x) + 0.5))
#endif


using std::max;
using std::min;
using std::floor;
using std::ceil;

#define M_PI       3.14159265358979323846   // pi

namespace roi_align_ratote_cpu {

	// forward
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
		T roi_center_h,
		T roi_center_w,
		T theta,
		std::vector<PreCalc<T>>& pre_calc) {
		int pre_calc_index = 0;
		T cosTheta = cos(theta);
		T sinTheta = sin(theta);
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

						// Rotate by theta around the center and translate
						T x = xx * cosTheta + yy * sinTheta + roi_center_w;
						T y = yy * cosTheta - xx * sinTheta + roi_center_h;

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

						// Save weights and indices
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
	void ROIAlignRotatedForward(
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
		DCHECK(roi_cols == 5 || roi_cols == 6);

		int n_rois = nthreads / channels / pooled_width / pooled_height;

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int n = 0; n < n_rois; n++) {
			int index_n = n * channels * pooled_width * pooled_height;
			// roi could have 5 or 6 columns
			const T* offset_bottom_rois = bottom_rois + n * roi_cols;
			int roi_batch_ind = 0;
			if (roi_cols == 6) {
				roi_batch_ind = offset_bottom_rois[0];
				offset_bottom_rois++;
			}

			// Do not round
			T roi_center_w = offset_bottom_rois[0] * spatial_scale;
			T roi_center_h = offset_bottom_rois[1] * spatial_scale;
			T roi_width = offset_bottom_rois[2] * spatial_scale;
			T roi_height = offset_bottom_rois[3] * spatial_scale;
			T theta = offset_bottom_rois[4] * M_PI / 180.0;

			// Force malformed ROIs to be 1x1
			roi_width = std::max(roi_width, (T)1.);
			roi_height = std::max(roi_height, (T)1.);
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

															 // We want to precalculate indices and weights shared by all channels,
															 // this is the key point of optimization.
			std::vector<PreCalc<T>> pre_calc(
				roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);

			// roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
			// Appropriate translation needs to be applied after.
			T roi_start_h = -roi_height / 2.0;
			T roi_start_w = -roi_width / 2.0;
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
				roi_center_h,
				roi_center_w,
				theta,
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
		} // if nchw
	}


}

namespace caffe {

	template <typename Dtype>
	void ROIAlignRatotePoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		ROIAlignRatoteParameter roi_align_ratote_param = this->layer_param_.roi_align_ratote_param();
		CHECK_GT(roi_align_ratote_param.pooled_h(), 0)
			<< "pooled_h must be > 0";
		CHECK_GT(roi_align_ratote_param.pooled_w(), 0)
			<< "pooled_w must be > 0";
		pooled_height_ = roi_align_ratote_param.pooled_h();
		pooled_width_ = roi_align_ratote_param.pooled_w();
		spatial_scale_ = roi_align_ratote_param.spatial_scale();
		LOG(INFO) << "Spatial scale: " << spatial_scale_;
	}

	template <typename Dtype>
	void ROIAlignRatotePoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		channels_ = bottom[0]->channels();
		height_ = bottom[0]->height();
		width_ = bottom[0]->width();
		top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,
			pooled_width_);
		/*max_idx_.Reshape(bottom[1]->num(), channels_, pooled_height_,
			pooled_width_);*/
	}

	template <typename Dtype>
	void ROIAlignRatotePoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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

		roi_align_ratote_cpu::ROIAlignRotatedForward<Dtype>(
			count,
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
	void ROIAlignRatotePoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		return;
	}


#ifdef CPU_ONLY
	STUB_GPU(ROIPoolingLayer);
#endif

	INSTANTIATE_CLASS(ROIAlignRatotePoolingLayer);
	REGISTER_LAYER_CLASS(ROIAlignRatotePooling);

}  // namespace caffe