#include <vector>
#include <algorithm>

#include "caffe/layers/depthwise_convolution.hpp"
#include "caffe/util/gpu_util.cuh"

#if !defined(_MSC_VER)
#define CUDA_UNROLL _Pragma("unroll")
#define CUDA_NOUNROLL _Pragma("nounroll")
#else
#define CUDA_UNROLL
#define CUDA_NOUNROLL
#endif


#define THREADS_PER_BLOCK          1024
#if __CUDA_ARCH__ >= 200
#define MY_KERNEL_MAX_THREADS  (2 * THREADS_PER_BLOCK)
#define MY_KERNEL_MIN_BLOCKS   3
#else
#define MY_KERNEL_MAX_THREADS  THREADS_PER_BLOCK
#define MY_KERNEL_MIN_BLOCKS   2
#endif

#define FULL_WARP_MASK 0xFFFFFFFF
#if CUDA_VERSION < 9000
template<typename DType>
__forceinline__ __device__ DType  __shfl_xor_sync(unsigned, DType val, int delta) {
	return __shfl_xor(val, delta);
}

template<typename DType>
__forceinline__ __device__ DType  __shfl_down_sync(unsigned, DType val, int delta) {
	return __shfl_down(val, delta);
}

// shuffle masks not used before CUDA 9.
#define CREATE_SHFL_MASK(mask, predicate) \
    unsigned mask = 0u;
#else
#define CREATE_SHFL_MASK(mask, predicate) \
    unsigned mask = __ballot_sync(FULL_WARP_MASK, (predicate))
#endif

#define MSHADOW_CUDA_POST_KERNEL_CHECK(x) \
  /* Code block avoids redefinition of cudaError_t err */ \
  do { \
    cudaError err = cudaPeekAtLastError(); \
    CHECK_EQ(err, cudaSuccess) << "Name: " << #x << " ErrStr:" << cudaGetErrorString(err); \
  } while (0)

namespace caffe {

	/*! \brief suggested thread number(logscale) for mapping kernel */
	const int kBaseThreadBits = 8;
	/*! \brief suggested thread number for mapping kernel */
	const int kBaseThreadNum = 1 << kBaseThreadBits;

	const int kMaxGridNum = 65535;

	template <typename Dtype>
	inline Dtype __device__ CudaMax(Dtype a, Dtype b) {
		return a > b ? a : b;
	}

	template <typename Dtype>
	inline Dtype __device__ CudaMin(Dtype a, Dtype b) {
		return a < b ? a : b;
	}


	template<typename Dtype>
	__global__ void
		/*__launch_bonds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)*/
		DepthwiseConv2dForwardKernel(
			const Dtype* input,
			const Dtype* filter,
			const DepthwiseArgs args,
			int num_outputs,
			Dtype* output,
			int kFilterHeight,
			int kFilterWidth) {
		const int in_channel = args.in_channel;
		const int in_height = args.in_height;
		const int in_width = args.in_width;
		const int filter_height = kFilterHeight > 0 ? kFilterHeight : args.filter_height;
		const int filter_width = kFilterWidth > 0 ? kFilterWidth : args.filter_width;
		const int stride_height = args.stride_height;
		const int stride_width = args.stride_width;
		const int pad_height = args.pad_height;
		const int pad_width = args.pad_width;
		const int out_channel = args.out_channel;
		const int out_height = args.out_height;
		const int out_width = args.out_width;

		CUDA_KERNEL_LOOP(thread_id, num_outputs) {
			// Compute the indexes of this thread in the output.
			//
			// We want coalesced reads so we make sure that each warp reads
			// a contiguous chunk of memory.
			//
			// THIS IS PROBABLY WRONG, we are not doing coalesced reads
			// into the input, because of the depth multiplier division...
			const int out_w = thread_id % out_width;
			const int out_h = (thread_id / out_width) % out_height;
			const int out_c = (thread_id / out_width / out_height) % out_channel;
			const int out_b = thread_id / out_width / out_height / out_channel;
			const int in_c = out_c;

			// Data is stored in the following format (let's assume we
			// flatten the height and width into one contiguous dimension
			// called "P".
			//
			// B1C1P1 B1C1P2 ..... B1C2P1 B1C2P2 ....
			// B2C1P1 B2C1P2 ..... B2C2P1 B2C2P2 ....
			//
			// Each row contains in_channel * in_height * in_width values
			// for each sample in the batch.
			//
			// We can further flatten it into:
			//
			// B1C1P1 B1C1P2 .....
			// B1C2P1 B1C2P2 ....
			// B2C1P1 B2C1P2 .....
			// B2C2P1 B2C2P2 ....
			//
			// where each row is a contiguous array of all of the spatial
			// pixels for a given batch and input depth.  The following
			// loop unrolls across the filter dimensions for a given thread,
			// indexing into the filter value and the corresponding input
			// patch.
			//
			// We can compute the index into the patch once right here.
			const int input_offset_temp = (out_b * in_channel + in_c) * (in_height * in_width);
			const int filter_offset_temp = in_c * filter_height * filter_width;

			// Finally, we can iterate over the spatial dimensions and perform the
			// convolution, writing into the output at the end.
			//
			// We perform an additional optimization, where we can determine
			// whether the patch fits within the image indices statically, and
			// avoid boundary checking within the loop.
			const int input_h_start = out_h * stride_height - pad_height;
			const int input_w_start = out_w * stride_width - pad_width;
			const int input_h_end = input_h_start + filter_height;
			const int input_w_end = input_w_start + filter_width;

			Dtype sum = 0;
			if (input_h_start >= 0 && input_w_start >= 0 &&
				input_h_end < in_height && input_w_end < in_width) {
				// Loop that doesn't need to check for boundary conditions.
				CUDA_UNROLL for (int f_h = 0; f_h < filter_height; ++f_h) {
					const int in_h = input_h_start + f_h;
					const int filter_offset_h = filter_width * f_h;
					CUDA_UNROLL for (int f_w = 0; f_w < filter_width; ++f_w) {
						const int in_w = input_w_start + f_w;
						const int input_offset = (input_offset_temp)+(in_h * in_width) + in_w;
						const int filter_offset = filter_offset_temp + filter_offset_h + f_w;
						sum += __ldg(input + input_offset) * __ldg(filter + filter_offset);
					}
				}
			}
			else {
				// Loop that needs to check for boundary conditions.
				CUDA_UNROLL for (int f_h = 0; f_h < filter_height; ++f_h) {
					const int in_h = input_h_start + f_h;
					const int filter_offset_h = filter_width * f_h;
					CUDA_UNROLL for (int f_w = 0; f_w < filter_width; ++f_w) {
						const int in_w = input_w_start + f_w;
						// TODO(vrv): the in_h check can be done outside of this loop;
						// benchmark both methods to determine the better decision.
						if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
							const int in_w = input_w_start + f_w;
							const int input_offset = input_offset_temp + (in_h * in_width) + in_w;
							const int filter_offset = filter_offset_temp + filter_offset_h + f_w;
							sum += __ldg(input + input_offset) * __ldg(filter + filter_offset);
						}
					}
				}
			}
			output[thread_id] = sum;
		}
	}

	// The DepthwiseConv2dKernelSmall perform either forward or backward input
	// convolution depending on a template argument of this enum.
	enum DepthwiseConv2dDirection { DIRECTION_FORWARD, DIRECTION_BACKWARD };

	// CUDA kernel to compute the depthwise convolution forward pass in NCHW format,
	// tailored for small images up to 32x32. Only use this kernel if
	// CanLaunchDepthwiseConv2dGPUSmall(args) returns true.
	// Tiles of the input and filter tensors are loaded into shared memory before
	// performing the convolution. Each thread handles two elements per iteration,
	// one each in the lower and upper half of a tile.
	// Backward input direction is the same as forward direction with the filter
	// rotated by 180бу.
	template <typename Dtype>
	__global__ /*__launch_bounds__(1024, 2)*/
		void DepthwiseConv2dKernelSmall(
			const DepthwiseArgs args,
			const Dtype* input,
			const Dtype* filter,
			Dtype* output,
			DepthwiseConv2dDirection kDirection,
			int kBlockSlices,
			bool kEvenHeight,
			int kFilterHeight,
			int kFilterWidth) {
		extern __shared__ __align__(sizeof(Dtype)) unsigned char shared_memory[];
		Dtype* const shared_data = reinterpret_cast<Dtype*>(shared_memory);

		const int in_height = args.in_height;
		const int in_width = args.in_width;
		const int in_channel = args.in_channel;
		const int filter_height = kFilterHeight > 0 ? kFilterHeight : args.filter_height;
		const int filter_width = kFilterWidth > 0 ? kFilterWidth : args.filter_width;
		const int pad_height = args.pad_height;
		const int pad_width = args.pad_width;

		// Fixed blockDim.z, tailored for maximum grid size for images of size 16x16.
		const int block_height = blockDim.y;

		// These values are the same for all threads and could
		// be precomputed on the CPU.
		const int block_pixels = in_width * block_height;
		const int block_size = block_pixels * kBlockSlices;
		const int in_pixels = in_width * in_height;
		const int in_increment = in_width - 1;
		const int filter_pixels = filter_height * filter_width;
		const int tile_width = in_width + filter_width - 1;
		const int even_height = kEvenHeight || (1 & ~in_height);
		const int tile_height = in_height + filter_height - even_height;
		const int tile_pixels = tile_width * tile_height;
		const int tile_size = tile_pixels * kBlockSlices;
		const int tile_offset = block_height * tile_width;
		const int pad_offset = pad_height * tile_width + pad_width;
		const int in_slices = in_channel * args.batch;
		const int in_blocks = (in_slices + kBlockSlices - 1) / kBlockSlices;

		const int thread_width = threadIdx.x;
		const int thread_height = threadIdx.y;
		const int thread_channel = threadIdx.z;

		// Position in block.
		const int thread_pix = thread_height * in_width + thread_width;
		const int thread_idx = thread_channel * block_pixels + thread_pix;

		// Initialize tile, in particular the padding.
		for (int i = thread_idx; i < tile_size; i += block_size) {
			shared_data[i] = Dtype(0);
		}
		__syncthreads();

		// Position in tensors.
		const int tensor_idx = thread_channel * in_pixels + thread_pix;

		// Position in (padded) shared memory.
		const int data_pix = thread_height * tile_width + thread_width;
		const int data_idx = thread_channel * tile_pixels + data_pix;

		// Position in shared memory, offset by pad_height / pad_width.
		const int tile_idx = data_idx + pad_offset;

		const int filter_pix = thread_pix;
		const int filter_channel = thread_channel;
		const int filter_idx = filter_pixels * filter_channel + filter_pix;

		const int max_slice = in_slices - thread_channel;
		const int filter_write_offset = filter_pix < filter_pixels ? tile_size + filter_idx : 0;
		const int filter_read_offset = tile_size +
			(kDirection == DIRECTION_FORWARD ?
				filter_pixels * filter_channel : filter_pixels * (filter_channel + 1));
		const bool skip_second = !kEvenHeight && thread_height + (in_height & 1) == block_height;

		for (int b = blockIdx.x; b < in_blocks; b += gridDim.x) {
			const int slice = b * kBlockSlices;

			const int inout_offset = slice * in_pixels + tensor_idx;
			const bool slice_in_range = slice < max_slice;

			if (slice_in_range) {
				const Dtype* const in_ptr = inout_offset + input;
				Dtype* const tile_ptr = tile_idx + shared_data;
				tile_ptr[0] = __ldg(in_ptr);
				if (!skip_second) {
					tile_ptr[tile_offset] = __ldg(block_pixels + in_ptr);
				}
			}

			if (filter_write_offset != 0) {
				const int filter_offset = ((slice + filter_channel) % in_channel)* filter_pixels + filter_pix;
				shared_data[filter_write_offset] = __ldg(filter_offset + filter);
			}

			// Note: the condition to reach this is uniform across the entire block.
			__syncthreads();

			if (slice_in_range) {
				Dtype sum1 = 0;
				Dtype sum2 = 0;
				int shared_offset = data_idx;
				const Dtype* filter_ptr = filter_read_offset + shared_data;
				CUDA_UNROLL for (int r = 0; r < filter_height; ++r) {
					CUDA_UNROLL for (int c = 0; c < filter_width; ++c) {
						if (kDirection == DIRECTION_BACKWARD) {
							filter_ptr--;
						}
						const Dtype filter_value = *filter_ptr;
						const Dtype* const tile_ptr = shared_offset + shared_data;
						sum1 += filter_value * tile_ptr[0];
						sum2 += filter_value * tile_ptr[tile_offset];
						++shared_offset;
						if (kDirection == DIRECTION_FORWARD) {
							filter_ptr++;
						}
					}
					shared_offset += in_increment;
				}
				Dtype* const out_ptr = inout_offset + output;
				if (kDirection == DIRECTION_FORWARD) {
					out_ptr[0] = sum1;
					if (!skip_second) {
						out_ptr[block_pixels] = sum2;
					}
				}
				else {
					out_ptr[0] += sum1;
					if (!skip_second) {
						out_ptr[block_pixels] += sum2;
					}
				}
			}

			// Note: the condition to reach this is uniform across the entire block.
			__syncthreads();
		}
	}

	template<typename Dtype>
	__global__ void /*__launch_bounds__(640, 2)*/
		DepthwiseConv2dBackwardDataKernel(
			const DepthwiseArgs args,
			const Dtype* out_grad,
			const Dtype* filter,
			Dtype* in_grad,
			int num_in_grad) {
		const int channel = args.in_channel;
		const int in_height = args.in_height;
		const int in_width = args.in_width;
		const int filter_height = args.filter_height;
		const int filter_width = args.filter_width;
		const int stride_height = args.stride_height;
		const int stride_width = args.stride_width;
		const int pad_height = args.pad_height;
		const int pad_width = args.pad_width;
		const int out_height = args.out_height;
		const int out_width = args.out_width;

		const int in_pixels = in_height * in_width;
		const int out_pixels = out_height * out_width;

		CUDA_KERNEL_LOOP(thread_id, num_in_grad) {
			// Compute the indexes of this thread in the input.
			const int in_w = thread_id % in_width;
			const int in_h = (thread_id / in_width) % in_height;
			const int channel_idx = (thread_id / in_width / in_height) % channel;
			const int batch_idx = thread_id / channel / in_width / in_height;
			Dtype sum = 0.0f;

			const int out_h_start = CudaMax<int>(
				0, (in_h - filter_height + pad_height + stride_height) / stride_height);
			const int out_h_end = CudaMin(
				out_height - 1, (in_h + pad_height) / stride_height);
			const int out_w_start = CudaMax<int>(
				0, (in_w - filter_width + pad_width + stride_width) / stride_width);
			const int out_w_end = CudaMin(
				out_width - 1, (in_w + pad_width) / stride_width);

			const int filter_offset_temp = channel_idx * filter_height * filter_width;
			const int out_grad_offset_temp = (batch_idx * channel * out_pixels) +
				(channel_idx * out_pixels);

			for (int out_h = out_h_start; out_h <= out_h_end; ++out_h) {
				const int f_h = in_h + pad_height - out_h * stride_height;
				const int filter_offset_h = filter_offset_temp + f_h * filter_width;
				const int out_grad_offset_h = out_grad_offset_temp + out_h * out_width;
				for (int out_w = out_w_start; out_w <= out_w_end; ++out_w) {
					const int f_w = in_w + pad_width - out_w * stride_width;
					const int filter_offset = filter_offset_h + f_w;
					const int out_grad_offset = out_grad_offset_h + out_w;
					sum += __ldg(out_grad + out_grad_offset) * __ldg(filter + filter_offset);
				}
			}
			const int in_grad_offset = (batch_idx * channel * in_pixels) +
				(channel_idx * in_pixels) + (in_h * in_width) + (in_w);
			in_grad[in_grad_offset] += sum;
		}
	}

	// A Cuda kernel to compute the depthwise convolution backprop w.r.t. filter.
	template <typename Dtype>
	__global__ void /*__launch_bounds__(640, 2)*/
		DepthwiseConv2dBackwardFilterKernel(
			const DepthwiseArgs args,
			const Dtype* out_backprop,
			const Dtype* input,
			Dtype* filter_backprop,
			int num_out_backprop,
			int kFilterWidth,
			int kFilterHeight) {
		const int in_channel = args.in_channel;
		const int in_height = args.in_height;
		const int in_width = args.in_width;
		const int filter_height = kFilterHeight > 0 ? kFilterHeight : args.filter_height;
		const int filter_width = kFilterWidth > 0 ? kFilterWidth : args.filter_width;
		const int stride_height = args.stride_height;
		const int stride_width = args.stride_width;
		const int pad_height = args.pad_height;
		const int pad_width = args.pad_width;
		const int out_channel = args.out_channel;
		const int out_height = args.out_height;
		const int out_width = args.out_width;

		CUDA_KERNEL_LOOP(thread_id, num_out_backprop) {
			// Compute the indexes of this thread in the output.
			const int out_w = thread_id % out_width;
			const int out_h = (thread_id / out_width) % out_height;
			const int out_c = (thread_id / out_width / out_height) % out_channel;
			const int out_b = thread_id / out_width / out_height / out_channel;
			const int in_c = out_c;

			// Decide if all input is valid, if yes, we can skip the boundary checks
			// for each input.
			const int in_row_start = out_h * stride_height - pad_height;
			const int in_col_start = out_w * stride_width - pad_width;
			const int in_row_end = in_row_start + filter_height;
			const int in_col_end = in_col_start + filter_width;

			const int out_backprop_offset =
				(out_b * out_channel * out_height * out_width) +
				(out_c * out_height * out_width) + (out_h * out_width) +
				(out_w);

			const Dtype out_bp = __ldg(out_backprop + out_backprop_offset);
			if (in_row_start >= 0 && in_col_start >= 0 &&
				in_row_end < in_height && in_col_end < in_width) {
				CUDA_UNROLL for (int f_h = 0; f_h < filter_height; ++f_h) {
					const int in_row = in_row_start + f_h;
					// Avoid repeated computation.
					const int input_offset_temp =
						(out_b * in_channel * in_height * in_width) +
						(in_c * in_height * in_width) + (in_row * in_width);
					const int filter_backprop_temp =
						(in_c * filter_width * filter_height) +
						(filter_width * f_h);

					CUDA_UNROLL for (int f_w = 0; f_w < filter_width; ++f_w) {
						int in_col = in_col_start + f_w;
						int input_offset = input_offset_temp + in_col;
						Dtype partial_sum = __ldg(input + input_offset) * out_bp;
						Dtype* addr = filter_backprop + (filter_backprop_temp + f_w);
						atomicAdd(addr, partial_sum);
					}
				}
			}
			else {
				CUDA_UNROLL for (int f_h = 0; f_h < filter_height; ++f_h) {
					const int in_row = in_row_start + f_h;
					// Avoid repeated computation.
					const int input_offset_temp =
						(out_b * in_channel * in_height * in_width) +
						(in_c * in_height * in_width) + (in_row * in_width);
					const int filter_backprop_temp =
						(in_c * filter_width * filter_height) +
						(filter_width * f_h);
					CUDA_UNROLL for (int f_w = 0; f_w < filter_width; ++f_w) {
						const int in_col = in_col_start + f_w;

						if (in_row >= 0 && in_row < in_height && in_col >= 0 && in_col < in_width) {
							const int input_offset = input_offset_temp + in_col;
							Dtype partial_sum = __ldg(input + input_offset) * out_bp;
							Dtype* addr = filter_backprop + (filter_backprop_temp + f_w);
							// Potentially many threads can add to the same address so we have
							// to use atomic add here.
							// TODO(jmchen): If atomic add turns out to be slow, we can:
							// 1. allocate multiple buffers for the gradients (one for each
							// example in a batch, for example). This can reduce the
							// contention on the destination; 2. Have each thread compute one
							// gradient for an element in the filters. This should work well
							// when the input depth is big and filter size is not too small.
							atomicAdd(addr, partial_sum);
						}
					}
				}
			}
		}
	}

	// CUDA kernel to compute the depthwise convolution backward w.r.t. filter in
	// NCHW format, tailored for small images up to 32x32. Only use this kernel if
	// CanLaunchDepthwiseConv2dGPUSmall(args) returns true.
	// Tiles of the input tensor are loaded into shared memory before performing the
	// convolution. Per iteration and filter element, each thread first performs
	// a partial convolution for two elements, one each in the lower and upper half
	// of a tile. The intermediate result of all pixels of a warp are then
	// accumulated and written to shared memory. Finally, the values in shared
	// memory are warp-accumulated (in chunks of kAccumPixels elements) and summed
	// up in global memory using atomics.
	// Requirements: threads per block must be multiple of 32 and <= launch_bounds,
	// kAccumPixels * 64 >= args.in_height * args.in_width * kBlockSlices.
	template <typename Dtype>
	__global__
		/*__launch_bounds__(1024, 2)*/
		void DepthwiseConv2dBackwardFilterKernelSmall(
			const DepthwiseArgs args,
			const Dtype* output,
			const Dtype* input,
			Dtype* filter,
			int kBlockSlices,
			int kAccumPixels,
			int kFilterHeight,
			int kFilterWidth) {
		extern __shared__ __align__(sizeof(Dtype)) unsigned char shared_memory[];
		Dtype* const shared_data = reinterpret_cast<Dtype*>(shared_memory);

		const int in_height = args.in_height;
		const int in_width = blockDim.x;  // slower (see b/62280718): args.in_width;
		const int in_channel = args.in_channel;
		const int filter_height = kFilterHeight > 0 ? kFilterHeight : args.filter_height;
		const int filter_width = kFilterWidth > 0 ? kFilterWidth : args.filter_width;
		const int pad_height = args.pad_height;
		const int pad_width = args.pad_width;

		const int block_height = blockDim.y;

		// These values are the same for all threads and could
		// be precomputed on the CPU.
		const int block_pixels = in_width * block_height;
		const int block_size = block_pixels * kBlockSlices;
		assert((block_size & 31) == 0);
		const int in_pixels = in_width * in_height;
		const int in_increment = in_width - 1;
		const int filter_pixels = filter_height * filter_width;
		const int tile_width = in_width + filter_width - 1;
		const int tile_height = 2 * block_height + filter_height - 1;
		const int tile_pixels = tile_width * tile_height;
		const int tile_size = tile_pixels * kBlockSlices;
		const int tile_offset = block_height * tile_width;
		const int pad_offset = pad_height * tile_width + pad_width;
		const int in_slices = in_channel * args.batch;
		const int in_blocks = (in_slices + kBlockSlices - 1) / kBlockSlices;
		// The accumulator has a fixed number of pixels that can be reduced by one
		// warp. Pixels beyond ceil(in_pixels * kBlockSlices / 64) are never written.
		assert(kAccumPixels * 64 >= in_height * in_width * kBlockSlices);
		const int accum_increment = kAccumPixels * kBlockSlices;
		const int accum_size = filter_pixels * accum_increment;

		const int thread_width = threadIdx.x;
		const int thread_height = threadIdx.y;
		const int thread_channel = threadIdx.z;

		// Position in block.
		const int thread_pix = thread_height * in_width + thread_width;
		const int thread_idx = thread_channel * block_pixels + thread_pix;

		// Initialize tile, in particular the padding and accumulator.
		for (int i = thread_idx; i < tile_size + accum_size; i += block_size) {
			shared_data[i] = Dtype(0);
		}
		__syncthreads();

		// Position in tensors.
		const int tensor_idx = thread_channel * in_pixels + thread_pix;

		// Position in (padded) shared memory.
		const int data_pix = thread_height * tile_width + thread_width;
		const int data_idx = thread_channel * tile_pixels + data_pix;

		// Position in shared memory, offset by pad_height / pad_width.
		const int tile_idx = data_idx + pad_offset;

		// Position in accumulator (kBlockSlices per warp, depth major).
		const int accum_pix = thread_pix / (32 / kBlockSlices);
		const int accum_idx = thread_channel * kAccumPixels + accum_pix;

		const int max_slice = in_slices - thread_channel;
		const int accum_offset = tile_size + accum_idx;
		const bool skip_second = block_height + thread_height >= in_height;

		for (int b = blockIdx.x; b < in_blocks; b += gridDim.x) {
			const int slice = b * kBlockSlices;

			const int inout_offset = slice * in_pixels + tensor_idx;
			const bool slice_in_range = slice < max_slice;

			if (slice_in_range) {
				const Dtype* const in_ptr = inout_offset + input;
				Dtype* const tile_ptr = tile_idx + shared_data;
				tile_ptr[0] = __ldg(in_ptr);
				if (!skip_second) {
					tile_ptr[tile_offset] = __ldg(block_pixels + in_ptr);
				}
			}

			// Note: the condition to reach this is uniform across the entire block.
			__syncthreads();

			// Not all threads of a warp may reach the __shfl_down_sync instruction
			// so we cannot use the FULL_WARP_MASK there
			CREATE_SHFL_MASK(active_threads, slice_in_range);

			if (slice_in_range) {
				const Dtype* const out_ptr = inout_offset + output;
				const Dtype out1 = __ldg(out_ptr);
				const Dtype out2 = skip_second ? Dtype(0) : __ldg(block_pixels + out_ptr);
				int shared_offset = data_idx;
				Dtype* accum_ptr = accum_offset + shared_data;
				CUDA_UNROLL for (int r = 0; r < filter_height; ++r) {
					CUDA_UNROLL for (int c = 0; c < filter_width; ++c) {
						const Dtype* const tile_ptr = shared_offset + shared_data;
						Dtype val = out1 * tile_ptr[0] + out2 * tile_ptr[tile_offset];
						// Warp-accumulate pixels of the same depth and write to accumulator.
						for (int delta = 16 / kBlockSlices; delta > 0; delta /= 2) {
							val += __shfl_down_sync(active_threads, val, delta);
						}
						if (!(thread_idx & 32 / kBlockSlices - 1)) {
							*accum_ptr = val;
						}
						++shared_offset;
						accum_ptr += accum_increment;
					}
					shared_offset += in_increment;
				}
			}

			// Note: the condition to reach this is uniform across the entire block.
			__syncthreads();

			const Dtype* const accum_data = tile_size + shared_data;
			for (int i = thread_idx; i < accum_size; i += block_size) {
				const int filter_idx = i / kAccumPixels;
				const int filter_pix = filter_idx / kBlockSlices;
				const int filter_channel = (slice + filter_idx % kBlockSlices) % in_channel;
				// convert to CHW
				const int filter_offset = filter_channel * filter_pixels +
					(filter_pix / filter_width) * filter_height + filter_pix % filter_width;

				if (filter_channel < in_channel) {
					Dtype val = accum_data[i];
					// Warp-accumulate pixels of the same depth from the accumulator.
					int lane_id;
					asm volatile ("mov.u32 %0, %laneid;" : "=r"(lane_id));
					int sub_warp = lane_id / kAccumPixels;
					int zeros = sub_warp * kAccumPixels;
					unsigned mask = (kAccumPixels == 32) ? FULL_WARP_MASK : (((1U << kAccumPixels) - 1) << zeros);
					for (int delta = kAccumPixels / 2; delta > 0; delta /= 2) {
						val += __shfl_xor_sync(mask, val, delta);
					}
					if (!(thread_idx & kAccumPixels - 1)) {
                                                //caffe_gpu_atomic_add<Dtype>(filter + filter_offset, val);
						atomicAdd(filter + filter_offset, val);
					}
				}
			}
		}
	}

	// Returns whether depthwise convolution forward or backward input pass can be
	// performed using the faster ('Small') variant of the kernel.
	bool CanLaunchDepthwiseConv2dGPUSmall(const DepthwiseArgs& args) {
		return args.stride_height == 1 && args.stride_width == 1 && args.in_height <= 32 &&
			args.in_width <= 32 && args.in_height == args.out_height &&
			args.in_width == args.out_width && args.pad_height >= 0 &&
			args.pad_height < args.filter_height && args.pad_width >= 0 &&
			args.pad_width < args.filter_width &&
			args.filter_height * args.filter_width <= (args.in_height + 1) / 2 * args.in_width;
	}

	// Returns whether depthwise convolution backward filter pass can be performed
	// using the faster ('Small') variant of the kernel.
	bool CanLaunchDepthwiseConv2dBackwardFilterGPUSmall(const DepthwiseArgs args,
		const int block_height) {
		return args.stride_height == 1 && args.stride_width == 1 && args.in_height <= 32 &&
			args.in_width <= 32 && args.in_height == args.out_height &&
			args.in_width == args.out_width && args.pad_height >= 0 &&
			args.pad_height < args.filter_height && args.pad_width >= 0 &&
			args.pad_width < args.filter_width && block_height <= args.in_height &&
			args.filter_height * args.filter_width <= block_height * args.in_width;
	}

	template <typename Dtype>
	void LaunchDepthwiseConv2dGPUSmall(
		cudaStream_t stream,
		const DepthwiseArgs args,
		const Dtype* input,
		const Dtype* filter,
		Dtype* output,
		DepthwiseConv2dDirection kDirection,
		int kBlockSlices,
		bool kEvenHeight) {
		const int block_height = (args.in_height + 1) / 2;
		dim3 block_dim = dim3(args.in_width, block_height, kBlockSlices);

		const int tile_width = args.in_width + args.filter_width - 1;
		const int tile_height = block_height * 2 + args.filter_height - 1;
		const int tile_pixels = tile_height * tile_width;
		const int filter_pixels = args.filter_height * args.filter_width;
		const int shared_memory_size =
			kBlockSlices * (tile_pixels + filter_pixels) * sizeof(Dtype);
		const int num_outputs =
			args.batch * args.out_height * args.out_width * args.out_channel;
		const int kMaxGridNum = 65535;
		int block_count = std::min(num_outputs / (block_dim.x * block_dim.y * block_dim.z) + 1,
			(unsigned)kMaxGridNum);
		//auto s = mshadow::Stream<mxnet::gpu>::GetStream(stream);
		if (args.filter_height == 3 && args.filter_width == 3) {
			DepthwiseConv2dKernelSmall<Dtype>
				<< <block_count, block_dim, shared_memory_size, stream >> > (
					args, input, filter, output, 
					kDirection, kBlockSlices, kEvenHeight, 3, 3);
		}
		else {
			DepthwiseConv2dKernelSmall<Dtype>
				<< <block_count, block_dim, shared_memory_size, stream >> > (
					args, input, filter, output, kDirection, 
					kBlockSlices, kEvenHeight, -1, -1);
		}
		MSHADOW_CUDA_POST_KERNEL_CHECK(DepthwiseConv2dKernelSmall);
	}


	template <typename Dtype>
	void LaunchDepthwiseConv2dGPUSmall(
		cudaStream_t stream,
		const DepthwiseArgs args,
		const Dtype* input,
		const Dtype* filter,
		Dtype* output,
		DepthwiseConv2dDirection kDirection,
		int kBlockSlices) {
		if (args.in_height & 1) {
			LaunchDepthwiseConv2dGPUSmall<Dtype>(
				stream, args, input, filter, output, kDirection, kBlockSlices, false);
		}
		else {
			LaunchDepthwiseConv2dGPUSmall<Dtype>(
				stream, args, input, filter, output, kDirection, kBlockSlices, true);
		}
	}

	template <typename Dtype>
	void LaunchDepthwiseConv2dGPUSmall(
		cudaStream_t stream,
		const DepthwiseArgs args,
		const Dtype* input,
		const Dtype* filter,
		Dtype* output,
		DepthwiseConv2dDirection kDirection) {
		// Maximize (power of two) kBlockSlices while keeping a block within 1024
		// threads (2 pixels per thread).
		const int block_pixels = (args.in_height + 1) / 2 * args.in_width;
		if (block_pixels > 256) {
			LaunchDepthwiseConv2dGPUSmall<Dtype>(stream, args, input, filter, output, kDirection, 2);
		}
		else if (block_pixels > 128) {
			LaunchDepthwiseConv2dGPUSmall<Dtype>(stream, args, input, filter, output, kDirection, 4);
		}
		else {
			LaunchDepthwiseConv2dGPUSmall<Dtype>(stream, args, input, filter, output, kDirection, 8);
		}
	}

	template <typename Dtype>
	bool TryLaunchDepthwiseConv2dBackwardFilterGPUSmall(
		cudaStream_t stream,
		const DepthwiseArgs args,
		const int block_height,
		const Dtype* out_grad,
		const Dtype* input,
		Dtype* filter_grad,
		int kBlockSlices,
		int kAccumPixels) {
		const int tile_width = args.in_width + args.filter_width - 1;
		const int tile_height = block_height * 2 + args.filter_height - 1;
		const int tile_pixels = tile_height * tile_width;
		const int filter_pixels = args.filter_height * args.filter_width;
		const int shared_memory_size =
			kBlockSlices * (tile_pixels + filter_pixels * kAccumPixels) * sizeof(Dtype);
		if (shared_memory_size > 46 * 1024) {
			return false;
		}

		dim3 block_dim = dim3(args.in_width, block_height, kBlockSlices);
		const int num_out_grad =
			args.batch * args.out_height * args.out_width * args.out_channel;
		int block_count = num_out_grad / (block_dim.x * block_dim.y * block_dim.z) + 1;

		if (args.filter_height == 3 && args.filter_width == 3) {
			DepthwiseConv2dBackwardFilterKernelSmall<Dtype>
				<< <block_count, block_dim, shared_memory_size, stream >> > (
					args, out_grad, input, filter_grad, kBlockSlices, kAccumPixels, 3, 3);
		}
		else {
			DepthwiseConv2dBackwardFilterKernelSmall<Dtype>
				<< <block_count, block_dim, shared_memory_size, stream >> > (
					args, out_grad, input, filter_grad, kBlockSlices, kAccumPixels, -1, -1);
		}
		MSHADOW_CUDA_POST_KERNEL_CHECK(DepthwiseConv2dBackwardFilterKernelSmall);
		return true;
	}

	template <typename Dtype>
	bool TryLaunchDepthwiseConv2dBackwardFilterGPUSmall(
		cudaStream_t stream,
		const DepthwiseArgs args,
		const int block_height,
		const Dtype* out_grad,
		const Dtype* input,
		Dtype* filter_grad,
		int kBlockSlices) {
		// Minimize (power of two) kAccumPixels, while satisfying
		// kAccumPixels * 32 >= block_height * in_width * kBlockSlices.
		const int block_pixels = block_height * args.in_width * kBlockSlices;
		if (block_pixels > 512) {
			return TryLaunchDepthwiseConv2dBackwardFilterGPUSmall<Dtype>(
				stream, args, block_height, out_grad, input, filter_grad, kBlockSlices, 32);
		}
		else if (block_pixels > 256) {
			return TryLaunchDepthwiseConv2dBackwardFilterGPUSmall<Dtype>(
				stream, args, block_height, out_grad, input, filter_grad, kBlockSlices, 16);
		}
		else {
			return TryLaunchDepthwiseConv2dBackwardFilterGPUSmall<Dtype>(
				stream, args, block_height, out_grad, input, filter_grad, kBlockSlices, 8);
		}
	}

	template <typename Dtype>
	bool TryLaunchDepthwiseConv2dBackwardFilterGPUSmall(
		cudaStream_t stream,
		const DepthwiseArgs args,
		const Dtype* out_grad,
		const Dtype* input,
		Dtype* filter_grad,
		DepthwiseConv2dDirection kDirection) {
		// Maximize (power of two) kBlockSlices while keeping a block within 1024
		// threads (2 pixels per thread).
		int block_slices = 8;
		int block_height = (args.in_height + 1) / 2;
		int round_mask = 1;
		for (; block_slices > 1; block_slices /= 2) {
			// args.in_width * block_height * kBlockSlices must be multiple of 32.
			for (; block_height * args.in_width * block_slices & 31;
				round_mask = round_mask * 2 + 1) {
				block_height = block_height + round_mask & ~round_mask;
			}
			int block_size = block_height * args.in_width * block_slices;
			if (block_size <= 1024) {
				break;
			}
		}

		if (!CanLaunchDepthwiseConv2dBackwardFilterGPUSmall(args, block_height)) {
			return false;
		}

		switch (block_slices) {
		case 8:
			return TryLaunchDepthwiseConv2dBackwardFilterGPUSmall<Dtype>(
				stream, args, block_height, out_grad, input, filter_grad, 8);
		case 4:
			return TryLaunchDepthwiseConv2dBackwardFilterGPUSmall<Dtype>(
				stream, args, block_height, out_grad, input, filter_grad, 4);
		case 2:
			return TryLaunchDepthwiseConv2dBackwardFilterGPUSmall<Dtype>(
				stream, args, block_height, out_grad, input, filter_grad, 2);
		default:
			return false;
		}
	}

	template<typename Dtype>
	void DepthwiseConv2dForwardGpu(
		cudaStream_t stream,
		const DepthwiseArgs& args,
		const Dtype *bottom_data,
		Dtype *top_data,
		const Dtype* weight,
		DepthwiseConv2dDirection kDirection) {

		// select kernel
		if (CanLaunchDepthwiseConv2dGPUSmall(args)) {
			LaunchDepthwiseConv2dGPUSmall<Dtype>(
				stream,
				args,
				bottom_data,
				weight,
				top_data,
				kDirection);
		}
		else {
			//int num_output = out_data[conv::kOut].shape_.Size();
			int num_output = args.out_channel * args.out_height * args.out_width;
			int block_num = num_output / kBaseThreadNum + 1 > kMaxGridNum ? kMaxGridNum : num_output / kBaseThreadNum + 1;

			if (args.filter_height == 3 && args.filter_width == 3) {
				DepthwiseConv2dForwardKernel<Dtype>
					<< <block_num, kBaseThreadNum, 0, stream >> > (
						bottom_data,
						weight,
						args,
						num_output,
						top_data, 3, 3);
			}
			else {
				DepthwiseConv2dForwardKernel<Dtype>
					<< <block_num, kBaseThreadNum, 0, stream >> > (
						bottom_data,
						weight,
						args,
						num_output,
						top_data, -1, -1);
			}
			MSHADOW_CUDA_POST_KERNEL_CHECK(DepthwiseConv2dForwardKernel);
		}
	}

	template<typename Dtype>
	void DepthwiseConv2dBackwardDataGpu(
		cudaStream_t stream,
		const DepthwiseArgs& args,
		const Dtype *bottom_diff,
		const Dtype *weight,
		Dtype *top_diff,
		DepthwiseConv2dDirection kDirection) {

		// select kernel
		if (CanLaunchDepthwiseConv2dGPUSmall(args)) {
			LaunchDepthwiseConv2dGPUSmall<Dtype>(
				stream,
				args,
				bottom_diff,
				weight,
				top_diff,
				kDirection);
		}
		else {
			int num_in_grad = args.out_channel * args.out_height * args.out_width;

			int block_num = num_in_grad / kBaseThreadNum + 1 > kMaxGridNum ? kMaxGridNum : num_in_grad / kBaseThreadNum + 1;
			DepthwiseConv2dBackwardDataKernel<Dtype>
				<< <block_num, kBaseThreadNum, 0, stream >> > (args,
					bottom_diff,
					weight,
					top_diff,
					num_in_grad);
			MSHADOW_CUDA_POST_KERNEL_CHECK(DepthwiseConv2dBackwardDataKernel);
		}
	}

	template<typename Dtype>
	void DepthwiseConv2dBackwardFilterGpu(
		cudaStream_t stream,
		const DepthwiseArgs& args,
		Dtype *top_diff,
		const Dtype *top_data,
		const Dtype *bottom_diff,
		DepthwiseConv2dDirection kDirection) {

		// select kernel
		if (TryLaunchDepthwiseConv2dBackwardFilterGPUSmall<Dtype>(
			stream, 
			args,
			bottom_diff,
			top_data,
			top_diff,
			kDirection)) {
			return;
		}
		else {

			int num_out_grad = args.out_channel * args.out_height * args.out_width;
			int block_num = args.out_channel * args.batch > kMaxGridNum ? kMaxGridNum : args.out_channel * args.batch;
			if (args.filter_width == 3 && args.filter_height == 3) {
				DepthwiseConv2dBackwardFilterKernel<Dtype>
					<< <block_num, kBaseThreadNum, 0, stream >> > (args,
						bottom_diff,
						top_data,
						top_diff,
						num_out_grad, 3, 3);
			}
			else {
				DepthwiseConv2dBackwardFilterKernel<Dtype>
					<< <block_num, kBaseThreadNum, 0, stream >> > (args,
						bottom_diff,
						top_data,
						top_diff,
						num_out_grad, -1, -1);
			}
			MSHADOW_CUDA_POST_KERNEL_CHECK(DepthwiseConv2dBackwardFilterKernel);
		}
	}

	template <typename Dtype>
	void DepthwiseConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		cudaStream_t stream = NULL;
		const Dtype *bottom_data = bottom[0]->gpu_data();
		Dtype *top_data = top[0]->mutable_gpu_data();
		const Dtype* weight = this->blobs_[0]->mutable_gpu_data();
		DepthwiseConv2dForwardGpu<Dtype>(stream, args, bottom_data, top_data, weight, DIRECTION_FORWARD);
		// bias forward
		if (this->bias_term_) {
		}
	}

	template <typename Dtype>
	void DepthwiseConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		cudaStream_t stream = NULL;
		Dtype *top_diff = top[0]->mutable_gpu_diff();
		const Dtype *top_data = top[0]->gpu_data();
		Dtype *weight = this->blobs_[0]->mutable_gpu_data();
		const Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
		DepthwiseConv2dBackwardDataGpu<Dtype>(stream,
			args,
			bottom_diff,
			top_data,
			top_diff,
			DIRECTION_BACKWARD);
		DepthwiseConv2dBackwardFilterGpu<Dtype>(
			stream,
			args,
			top_diff,
			top_data,
			bottom_diff,
			DIRECTION_BACKWARD);
		// bias backward
		if (this->bias_term_) {
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(DepthwiseConvolutionLayer);
}
