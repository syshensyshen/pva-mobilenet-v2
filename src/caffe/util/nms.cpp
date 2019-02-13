#include "caffe/util/nms.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
static
Dtype iou(const Dtype A[], const Dtype B[])
{
  if (A[0] > B[2] || A[1] > B[3] || A[2] < B[0] || A[3] < B[1]) {
    return 0;
  }

  // overlapped region (= box)
  const Dtype x1 = std::max(A[0],  B[0]);
  const Dtype y1 = std::max(A[1],  B[1]);
  const Dtype x2 = std::min(A[2],  B[2]);
  const Dtype y2 = std::min(A[3],  B[3]);

  // intersection area
  const Dtype width = std::max((Dtype)0,  x2 - x1 + (Dtype)1);
  const Dtype height = std::max((Dtype)0,  y2 - y1 + (Dtype)1);
  const Dtype area = width * height;

  // area of A, B
  const Dtype A_area = (A[2] - A[0] + (Dtype)1) * (A[3] - A[1] + (Dtype)1);
  const Dtype B_area = (B[2] - B[0] + (Dtype)1) * (B[3] - B[1] + (Dtype)1);

  // IoU
  return area / (A_area + B_area - area);
}

//template static float iou(const float A[], const float B[]);
//template static double iou(const double A[], const double B[]);

template float iou(const float A[], const float B[]);
template double iou(const double A[], const double B[]);

template <typename Dtype>
void nms_cpu(const int num_boxes,
             const Dtype boxes[],
             int index_out[],
             int* const num_out,
             const int base_index,
             const Dtype nms_thresh, const int max_num_out)
{
  int count = 0;
  std::vector<char> is_dead(num_boxes);
  for (int i = 0; i < num_boxes; ++i) {
    is_dead[i] = 0;
  }

  for (int i = 0; i < num_boxes; ++i) {
    if (is_dead[i]) {
      continue;
    }

    index_out[count++] = base_index + i;
    if (count == max_num_out) {
      break;
    }

    for (int j = i + 1; j < num_boxes; ++j) {
      if (!is_dead[j] && iou(&boxes[i * 5], &boxes[j * 5]) > nms_thresh) {
        is_dead[j] = 1;
      }
    }
  }

  *num_out = count;
  is_dead.clear();
}

template
void nms_cpu(const int num_boxes,
             const float boxes[],
             int index_out[],
             int* const num_out,
             const int base_index,
             const float nms_thresh, const int max_num_out);
template
void nms_cpu(const int num_boxes,
             const double boxes[],
             int index_out[],
             int* const num_out,
             const int base_index,
             const double nms_thresh, const int max_num_out);


template <typename Dtype>
void sort_cpu_box(Dtype list_cpu[], const int start, const int end,
const int num_top)
{
	const Dtype pivot_score = list_cpu[start * 5 + 4];
	int left = start + 1, right = end;
	Dtype temp[5];
	while (left <= right) {
		while (left <= end && list_cpu[left * 5 + 4] >= pivot_score) ++left;
		while (right > start && list_cpu[right * 5 + 4] <= pivot_score) --right;
		if (left <= right) {
			for (int i = 0; i < 5; ++i) {
				temp[i] = list_cpu[left * 5 + i];
			}
			for (int i = 0; i < 5; ++i) {
				list_cpu[left * 5 + i] = list_cpu[right * 5 + i];
			}
			for (int i = 0; i < 5; ++i) {
				list_cpu[right * 5 + i] = temp[i];
			}
			++left;
			--right;
		}
	}

	if (right > start) {
		for (int i = 0; i < 5; ++i) {
			temp[i] = list_cpu[start * 5 + i];
		}
		for (int i = 0; i < 5; ++i) {
			list_cpu[start * 5 + i] = list_cpu[right * 5 + i];
		}
		for (int i = 0; i < 5; ++i) {
			list_cpu[right * 5 + i] = temp[i];
		}
	}

	if (start < right - 1) {
		sort_cpu_box(list_cpu, start, right - 1, num_top);
	}
	if (right + 1 < num_top && right + 1 < end) {
		sort_cpu_box(list_cpu, right + 1, end, num_top);
	}
}

template
void sort_cpu_box(float list_cpu[], const int start, const int end,
const int num_top);

template
void sort_cpu_box(double list_cpu[], const int start, const int end,
const int num_top);


template<typename Dtype>
void swapbbox(const Dtype A[], const Dtype B[]){
    Dtype *tmp = (Dtype *)malloc(sizeof(Dtype) * 5);
    mmcpy(tmp, A,   sizeof(Dtype) * 5);
    mmcpy(B,   A,   sizeof(Dtype) * 5);
    mmcpy(A,   tmp, sizeof(Dtype) * 5);
	free(tmp);
}

template static void swapbbox(const float  A[], const float  B[]);
template static void swapbbox(const double A[], const double B[]);

template <typename Dtype>
void soft_nms_cpu(const int num_boxes,
             const Dtype boxes[],
             const Dtype nms_thresh, 
             int nms_method, Dtype sigma)
{
  for (int i = 0; i < num_boxes; ++i) {
    Dtype weight = (Dtype)1.0;
    Dtype max_score = boxes[i * 5 + 4];
    max_pos = i;    
    cur_pos = i + 1;

    while (cur_pos < num_boxes) {
        if (max_score < bboxes[cur_pos * 5 + 4]) {
            max_score = bboxes[cur_pos * 5 + 4];
            max_pos = cur_pos;
        }
        cur_pos++;
    }
    if (i != max_pos){
        swapbbox(bboxes+i*5, bboxes+max_pos*5);
    }
    cur_pos = i + 1;
    while (cur_pos < num_boxes){
        Dtype iou_v = iou(&boxes[i * 5], &boxes[cur_pos * 5]);
        if (0 == nms_method){
            weight = exp(-(iou_v * iou_v) / sigma);
        }
        else{
            if (iou_v > nms_thresh) 
                weight = 1 - iou_v;
        }
        bboxes[j*5 + 4] *= weight;
        if (boxes[j*5+4] < Dtype(0.001)){
            swapbbox(bboxes+j*5, bboxes+(num_boxes - 1) * 5);
            num_boxes -= 1;
            cur_pos -= 1;
        }
    }
}

template
void soft_nms_cpu(const int num_boxes,
             const float boxes[],
             const float nms_thresh, 
             int nms_method, float sigma);
template
void soft_nms_cpu(const int num_boxes,
             const double boxes[],
             const double nms_thresh, 
             int nms_method, double sigma);

}  // namespace caffe
