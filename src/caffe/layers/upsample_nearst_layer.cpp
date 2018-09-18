#include "caffe/layers/upsample_nearst_layer.hpp"

namespace caffe {

	int translate_idx(int index, int channels, int height, int width, int scale_factor) {
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

	int translate_idx_inv(
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

	template <typename Dtype>
	void UpsampleNearstLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *>& bottom, const vector<Blob<Dtype> *>& top) {
		UpsampleNearstParameter upsample_nearst_param = this->layer_param_.upsample_nearst_param();
		if (upsample_nearst_param.has_spatial_scale()) {
			spatial_scale_ = upsample_nearst_param.spatial_scale();
		}
		else {
			spatial_scale_ = 1.0f;
		}
	}

	template <typename Dtype>
	void UpsampleNearstLayer<Dtype>::Reshape(const vector<Blob<Dtype> *>& bottom, const vector<Blob<Dtype> *>& top) {
		int batch = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();

		int out_height = height * spatial_scale_;
		int output_width = width * spatial_scale_;
		top[0]->Reshape(batch, channels, out_height, output_width);
	}

	/*template <typename Dtype>
	Dtype UpsampleNormalization(int count, const Dtype * top_diff) {
		Dtype normlzation = 0.0;
		for (size_t i = 0; i < count; i++) {
			normlzation += top_diff[i] * top_diff[i];
		}
		return normlzation;
	}*/

	template <typename Dtype>
	void UpsampleNearstLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *>& bottom, const vector<Blob<Dtype> *>& top) {
		int batch = top[0]->num();
		int channels = top[0]->channels();
		int height = top[0]->height();
		int width = top[0]->width();

		const Dtype * bottom_data = bottom[0]->cpu_data();
		Dtype * top_data = top[0]->mutable_cpu_data();
		int scale_factor = spatial_scale_;
		caffe_memset(top[0]->count(), Dtype(0.0), top_data);
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < channels; c++) {
				for (int h = 0; h < height; h++) {
					for (int w = 0; w < width; w++) {
						int index = translate_idx(n * c * h * width + w, channels, height, width, scale_factor);
						top_data[n * c * h * width + w] = bottom_data[index];
					}
				}
			}
		}
	}

	template <typename Dtype>
	void UpsampleNearstLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype> *>& bottom) {
		int batch = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();

		Dtype * bottom_diff = bottom[0]->mutable_cpu_diff();
		const Dtype * top_diff = top[0]->cpu_diff();
		int count = bottom[0]->count();
		caffe_memset(count, Dtype(0.0), bottom_diff);
		Dtype normalization = caffe_cpu_l2norm(top[0]->count(), top_diff);
		int scale_factor = spatial_scale_;
		for (int n = 0; n < batch; n++) {
			for (int c = 0; c < channels; c++) {
				for (int h = 0; h < height; h++) {
					for (int w = 0; w < width; w++) {
						for (int i = 0; i < scale_factor; i++) {
							for (int j = 0; j < scale_factor; j++) {
								int index = translate_idx_inv(n *c *h * width + w, channels, height, width, scale_factor, i, j);
								bottom_diff[n *c *h * width + w] += top_diff[index] / normalization;
							}
						}
					}
				}
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(UpsampleNearstLayer);
#endif

	INSTANTIATE_CLASS(UpsampleNearstLayer);
	REGISTER_LAYER_CLASS(UpsampleNearst);
}