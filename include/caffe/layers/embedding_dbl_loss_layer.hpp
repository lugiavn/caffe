#ifndef CAFFE_EMBEDDING_DBL_LOSS_LAYER_HPP_
#define CAFFE_EMBEDDING_DBL_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the embedding loss
 * @param bottom input Blob vector (length 3)
 * @param top output Blob vector (length 1)
 */
template <typename Dtype>
class EmbeddingDBLLossLayer : public LossLayer<Dtype> {
 public:
  explicit EmbeddingDBLLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline const char* type() const { return "EmbeddingDBLLoss"; }
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 2;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
/*
  Blob<Dtype> diff_;  // cached for backward pass
  Blob<Dtype> dist_sq_;  // cached for backward pass
  Blob<Dtype> diff_sq_;  // tmp storage for gpu forward pass
  Blob<Dtype> summer_vec_;  // tmp storage for gpu forward pass
  */

   vector< vector<Dtype> > distances_;
   bool _triplet;
   bool _dbl;
   bool _exhausting;
};

}  // namespace caffe

#endif  // CAFFE_EMBEDDING_DBL_LOSS_LAYER_HPP_
