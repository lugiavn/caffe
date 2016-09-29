#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_data_with_rotation_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#define DEGREE_2_RAD(x) ((x)/180.0*3.1415)
#define RESIZE_OPS cv::INTER_CUBIC

namespace caffe {

// First, do random rotation
// Then random scale
// Then crop center
// Then resize to newsize
cv::Mat ApplyAdditionalImageTransform(const cv::Mat & cv_img_origin, 
      float random_rotation = 0, float random_scale = 1, float crop_center = 0,
      int width = 0, int height = 0, float * out_rotation_angle = NULL) {

    if (random_rotation == 0) {
        cv::Mat cv_img4;
        if (height > 0 && width > 0) {
          cv::resize(cv_img_origin, cv_img4, cv::Size(width, height), 0, 0, RESIZE_OPS);
        } else {
          cv_img4 = cv_img_origin;
        }
        return cv_img4;
    }

    // random rotation
    cv::Mat cv_img1;
    if (random_rotation == 0)
      cv_img1 = cv_img_origin;
    else
    {
	    float angle = (rand() % int(random_rotation)) - random_rotation/2.0;
	    int len = std::max(cv_img_origin.cols, cv_img_origin.rows);
	    cv::Point2f pt(len/2., len/2.);
	    cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
	    cv::warpAffine(cv_img_origin, cv_img1, r, cv::Size(len, len), RESIZE_OPS);
	    if (out_rotation_angle)
		    *out_rotation_angle = angle;
    }

    // random scale
    cv::Mat cv_img2;
    if (random_scale == 1 || random_scale < 0)
      cv_img2 = cv_img1;
    else
    {
	    float s = (rand() % 1000) / 999.0;
	    if (rand() % 2 > 0)
		s = s * 1.0 + (1 - s) * random_scale;
	    else
		s = s * 1.0 + (1 - s) * 1.0 / random_scale;
            //printf("Resize of factor %f from %f \n", s, random_scale);
        cv::Size size(cv_img1.cols * s, cv_img1.rows * s);
        cv::resize(cv_img1, cv_img2, size, 0, 0, RESIZE_OPS);
    }

    // crop center
    cv::Mat cv_img3 = cv_img2;
    if (crop_center > 0)
    {
      cv::Rect myROI(cv_img2.cols/2.0 - crop_center/2.0, cv_img2.rows/2.0 - crop_center/2.0, crop_center, crop_center);
      cv::Mat cv_img3 = cv_img2(myROI);
    }

    // resize
    cv::Mat cv_img4;
    if (height > 0 && width > 0) {
      cv::resize(cv_img3, cv_img4, cv::Size(width, height), 0, 0, RESIZE_OPS);
    } else {
      cv_img4 = cv_img3;
    }

    // viz
/*
    if (0) {
        cv::namedWindow("a");
        cv::imshow("a", cv_img_origin);
        cv::namedWindow("b");
        cv::imshow("b", cv_img4);
        cv::waitKey();
    }
*/

    return cv_img4;
}

template <typename Dtype>
ImageDataWithRotationLayer<Dtype>::~ImageDataWithRotationLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageDataWithRotationLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  caffe::ImageDataWithRotationParameter d = this->layer_param_.image_data_with_rotation_param();
  const bool fix_shuffle = d.fix_shuffle();
  const int line_start = d.line_start();
  const int line_end = d.line_end();
  const float random_rotation = d.random_rotation();
  const int crop_center_size = d.crop_center_size();
  const float random_scale = d.random_scale();
  const bool output_rotation = d.output_rotation();
  const bool mine_pairs_with_uniform_class = d.mine_pairs_with_uniform_class();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  size_t pos;
  int label;
  int count_line = 0;
  while (std::getline(infile, line)) {
    pos = line.find_last_of(' ');
    pos = line.find_first_of(' ');
    label = atoi(line.substr(pos + 1).c_str());
    if (count_line >= line_start && (count_line <= line_end || line_end < 0))
      lines_.push_back(std::make_pair(line.substr(0, pos), label));
    count_line++;
  }

  CHECK(!lines_.empty()) << "File is empty";

  // index classes
  if (mine_pairs_with_uniform_class)
  {
    CHECK(!this->layer_param_.image_data_param().shuffle());
    int class_num = 0;
    for (int i = 0; i < lines_.size(); i++)
        class_num = std::max(class_num, 1 + lines_[i].second);
    for (int i = 0; i < class_num; i++)
        lines_from_class_.push_back(vector<int>());
    for (int i = 0; i < lines_.size(); i++)
        lines_from_class_[lines_[i].second].push_back(i);
    srand (time(NULL));
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = fix_shuffle ? 54635 : caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    0, 0, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  cv_img = ApplyAdditionalImageTransform(cv_img, random_rotation, random_scale, crop_center_size, new_width, new_height);
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  if (random_rotation > 0 && output_rotation)
    label_shape.push_back(3);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void ImageDataWithRotationLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageDataWithRotationLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  caffe::ImageDataWithRotationParameter d = this->layer_param_.image_data_with_rotation_param();
  const float random_rotation = d.random_rotation();
  const int crop_center_size = d.crop_center_size();
  const float random_scale = d.random_scale();
  const bool output_rotation = d.output_rotation();
  const bool mine_pairs_with_uniform_class = d.mine_pairs_with_uniform_class();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  int line1 = -1, line2 = -1;
  for (int item_id = 0; item_id < batch_size; ++item_id) {

    // set lines_id_ to form pairs
    if (mine_pairs_with_uniform_class)
    {
        // generate a random pair
        if (item_id % 2 == 0)
        {
            // sample class
            int class_id = rand() % lines_from_class_.size();
            while (lines_from_class_[class_id].size() < 2)
                class_id = rand() % lines_from_class_.size();

            // sample pair
            line1 = lines_from_class_[class_id][rand() % lines_from_class_[class_id].size()];
            line2 = lines_from_class_[class_id][rand() % lines_from_class_[class_id].size()];
            while (line1 == line2)
                line2 = lines_from_class_[class_id][rand() % lines_from_class_[class_id].size()];

            //printf("Sample class %d, line %d %d, class %d %d \n", class_id, line1, line2, lines_[line1].second, lines_[line1].second);
        }

        // set the 1st of the pair
        if (item_id % 2 == 0)
            lines_id_ = line1;

        // set the 2nd of the pair
        if (item_id % 2 == 1)
            lines_id_ = line2;
    }

    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        0, 0, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    float angle;
    cv_img = ApplyAdditionalImageTransform(cv_img, random_rotation, random_scale, crop_center_size, new_width, new_height, &angle);
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    if (random_rotation == 0 || !output_rotation)
      prefetch_label[item_id] = lines_[lines_id_].second;
    else
    {
        prefetch_label[3*item_id] = lines_[lines_id_].second;
        prefetch_label[3*item_id + 1] = sin(DEGREE_2_RAD(angle));
        prefetch_label[3*item_id + 2] = cos(DEGREE_2_RAD(angle));
    }
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageDataWithRotationLayer);
REGISTER_LAYER_CLASS(ImageDataWithRotation);

}  // namespace caffe
#endif  // USE_OPENCV
