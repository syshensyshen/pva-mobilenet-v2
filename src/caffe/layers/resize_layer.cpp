#include "caffe/layers/resize_layer.hpp"

namespace caffe {
	template <typename Dtype>
	void Resize2xForward(
		int batch_size,
		int num_channels,
		int input_height,
		int input_width,
		const Dtype* bottom_data,
		Dtype* top_data) {
		const int output_height = input_height << 1;
		const int output_width = input_width << 1;
		for (int n = 0; n < batch_size; ++n) {
			for (int c = 0; c < num_channels; ++c) {
				for (int y = 0; y < output_height; ++y) {
					const int in_y = (y >> 1);

					for (int x = 0; x < input_width; ++x) {
						const Dtype v = bottom_data[in_y * input_width + x];
						const int oidx = output_width * y + (x << 1);
						top_data[oidx + 0] = v;
						top_data[oidx + 1] = v;
					}
				}
				bottom_data += input_height * input_width;
				top_data += output_height * output_width;
			}
		}
	}

	template <typename Dtype>
	void ResizeNxForward(
		int batch_size,
		int channels,
		int input_height,
		int input_width,
		const Dtype* bottom_data,
		Dtype* top_data,
		Dtype height_scale,
		Dtype width_scale
		) {
		const int output_height = input_height * height_scale;
		const int output_width = input_width * width_scale;
		for (int n = 0; n < batch_size; ++n) {
			for (int c = 0; c < channels; ++c) {
				for (int y = 0; y < output_height; ++y) {
					const int in_y = std::min((int)(y / height_scale), (input_height - 1));
					for (int x = 0; x < input_width; ++x) {
						const int in_x = std::min((int)(x / width_scale), (input_width - 1));
						top_data[output_height * y + x] = bottom_data[input_height * y + in_x];
					}
				}
				bottom_data += input_height * input_width;
				top_data += output_height * output_width;
			}
		}
	}

	template <typename Dtype>
	void ResizeBackward(
		int batch_size,
		int channels,
		int input_height,
		int input_width,
		long int count,
		Dtype *bottom_diff,
		const Dtype *top_diff,
		Dtype height_scale,
		Dtype width_scale,
		Dtype normlzation
	    ) {
		const int output_height = input_height * height_scale;
		const int output_width = input_width * width_scale;
		for (int n = 0; n < batch_size; ++n) {
			for (int c = 0; c < channels; ++c) {
				for (int y = 0; y < input_height; ++y) {
					const int out_y = std::min((int)(y / height_scale),
						(output_height - 1));
					for (int x = 0; x < input_width; ++x) {
						const int out_x = std::min((int)(x / width_scale),
							(output_width - 1));
						bottom_diff[output_width * out_y + out_x] += top_diff[input_width * y + x] / normlzation;
					}
				}
				top_diff += input_height * input_width;
				bottom_diff += output_height * output_width;
			}
		}

	}

	template <typename Dtype>
	void ResizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *>& bottom, const vector<Blob<Dtype> *>& top) {

		ResizeLayerParameter resize_param = this->layer_param_.resize_layer_param();
		scale_h_ = resize_param.has_scale_h() ? resize_param.scale_h() : 1.0f;
		scale_w_ = resize_param.has_scale_w() ? resize_param.scale_w() : 1.0f;

	}

	template <typename Dtype>
	void ResizeLayer<Dtype>::Reshape(const vector<Blob<Dtype> *>& bottom, const vector<Blob<Dtype> *>& top) {

		int batch = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();

		int out_height = height * scale_h_;
		int output_width = width * scale_w_;
		top[0]->Reshape(batch, channels, out_height, output_width);

	}

	template <typename Dtype>
	Dtype ResizeNormalization(int count, const Dtype * top_diff) {
		Dtype normlzation = 0;
		for (size_t i = 0; i < count; i++) {
			normlzation += top_diff[i] * top_diff[i];
		}
		return normlzation;
	}

	template <typename Dtype>
	void ResizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *>& bottom, const vector<Blob<Dtype> *>& top) {

		int batch = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
		const Dtype *bottom_data = bottom[0]->cpu_data();
		Dtype *top_data = top[0]->mutable_cpu_data();
		caffe_memset(top[0]->count(), Dtype(0.0), top_data);
		if (scale_w_ == 2.0f && scale_h_ == 2.0f) {
			Resize2xForward(
				batch, channels, height, width, bottom_data, top_data);
		}
		else {
			ResizeNxForward(batch, channels, height, width, bottom_data, top_data, scale_h_, scale_w_);
		}
	}

	template <typename Dtype>
	void ResizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype> *>& bottom) {
		int batch = top[0]->num();
		int channels = top[0]->channels();
		int height = top[0]->height();
		int width = top[0]->width();
		const Dtype * top_diff = top[0]->cpu_diff();
		Dtype * bottom_diff = bottom[0]->mutable_cpu_diff();
		int count = top[0]->count();
		//Dtype normization = ResizeNormalization(count, top_diff);
		Dtype normization = caffe_cpu_l2norm(top[0]->count(), top[0]->cpu_data());
		ResizeBackward(batch, channels, height, width, count, bottom_diff, top_diff, scale_h_, scale_w_, normization);

	}
#ifdef CPU_ONLY
	STUB_GPU(ResizeLayer);
#endif

	INSTANTIATE_CLASS(ResizeLayer);
	REGISTER_LAYER_CLASS(Resize);

}