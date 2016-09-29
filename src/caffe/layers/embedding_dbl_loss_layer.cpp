#include <algorithm>
#include <vector>

#include "caffe/layers/embedding_dbl_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

using namespace std;
template <typename Dtype>
Dtype SquaredDistance(const Dtype * a, const Dtype * b, int c);
float get_best_accurcy(vector<float> pos_distances, vector<float> neg_distances, float * threshold = NULL);
float get_triplet_prob(float match_distance, float nonmatch_distance);
float get_triplet_dbl_log_loss(float match_distance, float nonmatch_distance);
template <typename Dtype>
void apply_triplet_dbl_grad(float p,
                            const Dtype* anchor_feature, Dtype* anchor_grad,
                            const Dtype* match_feature, Dtype* match_grad,
                            const Dtype* nonmatch_feature, Dtype* nonmatch_grad,
                            int channels);
float get_pair_prob(float d, float m);
float get_pair_dbl_log_loss(float d,  int label, float m);
template <typename Dtype>
void apply_pair_dbl_grad(float d, float m, int label,
                         const Dtype* a_feature, Dtype* a_grad,
                         const Dtype* b_feature, Dtype* b_grad, int channels, float factor = 1);


namespace caffe {

template <typename Dtype>
void EmbeddingDBLLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK(bottom.size() == 2 || bottom.size() == 3);

  bool pair_input = (bottom.size() == 3);
  if (pair_input)
  {
      CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
      CHECK_EQ(bottom[1]->height(), 1);
      CHECK_EQ(bottom[1]->width(), 1);
      CHECK_EQ(bottom[2]->channels(), 1);
      CHECK_EQ(bottom[2]->height(), 1);
      CHECK_EQ(bottom[2]->width(), 1);
  } else
  {
      CHECK_EQ(bottom[1]->channels(), 1);
      CHECK_EQ(bottom[1]->height(), 1);
      CHECK_EQ(bottom[1]->width(), 1);
  }
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);

  int count = bottom[0]->shape(0);
  for (int i = 0; i < count; i++)
      distances_.push_back(vector<Dtype>(count, 0));

  string description = this->layer_param_.embedding_dbl_loss_param().description();
  _triplet = strstr(description.c_str(), "triplet") != NULL;
  _dbl = strstr(description.c_str(), "dbl") != NULL;
  _exhausting = strstr(description.c_str(), "exhausting") != NULL;
}

template <typename Dtype>
void EmbeddingDBLLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); i++)
      CHECK_EQ(bottom[0]->shape(0), bottom[i]->shape(0))
          << "The inputs to this loss layer should have the same first dimension.";
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
  for (int i = 1; i < top.size(); i++)
      top[i]->Reshape(loss_shape);
}

template <typename Dtype>
void EmbeddingDBLLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    int count = bottom[0]->shape(0);
    int channels = bottom[0]->channels();

    bool pair_input = (bottom.size() == 3);

    int example_count = 0;

    // pre-compute all pairwise distances
    for (int i = 0; i < count; i++)
      for (int j = 0; j < count; j++)
      {
          const Dtype* a = bottom[0]->cpu_data() + i*channels;
          const Dtype* b = bottom[pair_input ? 1 : 0]->cpu_data() + j*channels;
          distances_[i][j] = SquaredDistance(a, b, channels);
      }

    vector<float> pos_distances, neg_distances;

    // Loop through the batch
    double loss = 0;
    if (pair_input)
    {
        // loop through every pair of pairs
        for (int i = 0; i < count; i++)
        for (int j = 0; j < count; j++)
        {
            // positive match
            int label = static_cast<int>(bottom[2]->cpu_data()[i]);
            if (label != 1)
                continue;
            if (i == j)
                continue;

            pos_distances.push_back(distances_[i][i]);
            pos_distances.push_back(distances_[j][j]);
            neg_distances.push_back(distances_[i][j]);
            neg_distances.push_back(distances_[j][i]);

            // triplet: pairs_i_0, pairs_i_1, pairs_j_1
            if (_triplet)
            {
                loss += get_triplet_dbl_log_loss(distances_[i][i], distances_[i][j]);
                example_count++;
            }
            else
            {
                loss += get_pair_dbl_log_loss(distances_[i][i], 1, 88);
                loss += get_pair_dbl_log_loss(distances_[i][j], 0, 88);
                example_count++;
                example_count++;
            }

            // triplet: pairs_i_1, pairs_i_0, pairs_j_0
            if (_triplet)
            {
                loss += get_triplet_dbl_log_loss(distances_[i][i], distances_[j][i]);
                example_count++;
            } else
            {
                loss += get_pair_dbl_log_loss(distances_[i][i], 1, 88);
                loss += get_pair_dbl_log_loss(distances_[j][i], 0, 88);
                example_count++;
                example_count++;
            }
        }

    } else
    {
        // loop through every triplets
        for (int i = 0; i < count; i++)
        for (int j = 0; j < count; j++)
        for (int k = 0; k < count; k++)
        {
            int labeli = static_cast<int>(bottom[1]->cpu_data()[i]);
            int labelj = static_cast<int>(bottom[1]->cpu_data()[j]);
            int labelk = static_cast<int>(bottom[1]->cpu_data()[k]);
            if (i != j && labeli == labelj && labeli != labelk)
            {
                pos_distances.push_back(distances_[i][j]);
                neg_distances.push_back(distances_[i][k]);

                if (_triplet)
                {
                    loss += get_triplet_dbl_log_loss(distances_[i][j], distances_[i][k]);
                    example_count++;
                } else
                {
                    loss += get_pair_dbl_log_loss(distances_[i][j], 1, 88);
                    loss += get_pair_dbl_log_loss(distances_[i][k], 0, 88);
                    example_count++;
                    example_count++;
                }
            }
        }
    }

    CHECK_GT(example_count, 0);
    top[0]->mutable_cpu_data()[0] = loss / example_count;

    // Extra outputs
    if (top.size() >= 2)
        top[1]->mutable_cpu_data()[0] = example_count;
    if (top.size() >= 3)
    {
        float threshold = 0;
        top[2]->mutable_cpu_data()[0] = get_best_accurcy(pos_distances, neg_distances, &threshold);
        if (top.size() >= 4)
            top[3]->mutable_cpu_data()[0] = threshold;
    }
}

template <typename Dtype>
void EmbeddingDBLLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    int count = bottom[0]->shape(0);
    int channels = bottom[0]->channels();

    bool pair_input = (bottom.size() == 3);
    int example_count = 0;

    // Reset diff
    bottom[0]->scale_diff(0);
    bottom[1]->scale_diff(0);

    // Loop through the batch
    if (pair_input)
    {
        // loop through every pair of pairs
        for (int i = 0; i < count; i++)
        for (int j = 0; j < count; j++)
        {
            // positive match
            int label = static_cast<int>(bottom[2]->cpu_data()[i]);
            if (label != 1)
                continue;
            if (i == j)
                continue;

            const Dtype* pairs_i_0 = bottom[0]->cpu_data() + i*channels;
            const Dtype* pairs_i_1 = bottom[1]->cpu_data() + i*channels;
            const Dtype* pairs_j_0 = bottom[0]->cpu_data() + j*channels;
            const Dtype* pairs_j_1 = bottom[1]->cpu_data() + j*channels;
            Dtype* grad_pairs_i_0 = bottom[0]->mutable_cpu_diff() + i*channels;
            Dtype* grad_pairs_i_1 = bottom[1]->mutable_cpu_diff() + i*channels;
            Dtype* grad_pairs_j_0 = bottom[0]->mutable_cpu_diff() + j*channels;
            Dtype* grad_pairs_j_1 = bottom[1]->mutable_cpu_diff() + j*channels;


            if (_triplet)
            {
                // triplet: pairs_i_0, pairs_i_1, pairs_j_1
                float p = get_triplet_prob(distances_[i][i], distances_[i][j]);
                apply_triplet_dbl_grad(p, pairs_i_0, grad_pairs_i_0, pairs_i_1, grad_pairs_i_1, pairs_j_1, grad_pairs_j_1, channels);
                example_count++;

                // triplet: pairs_i_1, pairs_i_0, pairs_j_0
                p = get_triplet_prob(distances_[i][i], distances_[j][i]);
                apply_triplet_dbl_grad(p, pairs_i_1, grad_pairs_i_1, pairs_i_0, grad_pairs_i_0, pairs_j_0, grad_pairs_j_0, channels);
                example_count++;

            } else
            {
                apply_pair_dbl_grad(distances_[i][i], 88, 1, pairs_i_0, grad_pairs_i_0, pairs_i_1, grad_pairs_i_1, channels);
                example_count++;
                apply_pair_dbl_grad(distances_[i][j], 88, 0, pairs_i_0, grad_pairs_i_0, pairs_j_1, grad_pairs_j_1, channels);
                example_count++;

                apply_pair_dbl_grad(distances_[i][i], 88, 1, pairs_i_1, grad_pairs_i_1, pairs_i_0, grad_pairs_i_0, channels);
                example_count++;
                apply_pair_dbl_grad(distances_[j][i], 88, 0, pairs_i_1, grad_pairs_i_1, pairs_j_0, grad_pairs_j_0, channels);
                example_count++;
            }
        }

    } else
    {
        // loop through every triplets
        for (int i = 0; i < count; i++)
        for (int j = 0; j < count; j++)
        for (int k = 0; k < count; k++)
        {
            int labeli = static_cast<int>(bottom[1]->cpu_data()[i]);
            int labelj = static_cast<int>(bottom[1]->cpu_data()[j]);
            int labelk = static_cast<int>(bottom[1]->cpu_data()[k]);
            if (i != j && labeli == labelj && labeli != labelk)
            {
                const Dtype* anchor_feature = bottom[0]->cpu_data() + i*channels;
                const Dtype* match_feature = bottom[0]->cpu_data() + j*channels;
                const Dtype* nonmatch_feature = bottom[0]->cpu_data() + k*channels;
                Dtype* anchor_grad = bottom[0]->mutable_cpu_diff() + i*channels;
                Dtype* match_grad = bottom[0]->mutable_cpu_diff() + j*channels;
                Dtype* nonmatch_grad = bottom[0]->mutable_cpu_diff() + k*channels;


                if (_triplet)
                {
                    float p = get_triplet_prob(distances_[i][j], distances_[i][k]);
                    apply_triplet_dbl_grad(p, anchor_feature, anchor_grad, match_feature, match_grad, nonmatch_feature, nonmatch_grad, channels);
                    example_count++;
                } else
                {
                    apply_pair_dbl_grad(distances_[i][j], 88, 1, anchor_feature, anchor_grad, match_feature, match_grad, channels);
                    example_count++;
                    apply_pair_dbl_grad(distances_[i][k], 88, 0, anchor_feature, anchor_grad, nonmatch_feature, nonmatch_grad, channels);
                    example_count++;
                }
            }
        }
    }

    CHECK_GT(example_count, 0);
    double loss_weight = top[0]->cpu_diff()[0]; // WTF
    bottom[0]->scale_diff(1.0 / example_count * loss_weight);
    bottom[1]->scale_diff(1.0 / example_count * loss_weight);
}

#ifdef CPU_ONLY
STUB_GPU(EmbeddingDBLLossLayer);
#endif

INSTANTIATE_CLASS(EmbeddingDBLLossLayer);
REGISTER_LAYER_CLASS(EmbeddingDBLLoss);

}  // namespace caffe

template <typename Dtype>
Dtype SquaredDistance(const Dtype * a, const Dtype * b, int c) {
    Dtype d = 0;
    for (int i = 0; i < c; i++)
        d += (a[i] - b[i]) * (a[i] - b[i]);
    CHECK(d >= 0 || d <= 0);
    return d;
}

float get_best_accurcy(vector<float> pos_distances, vector<float> neg_distances, float * threshold)
{
    std::sort(pos_distances.begin(), pos_distances.end());
    std::sort(neg_distances.begin(), neg_distances.end());
    int best_correct_count = neg_distances.size();
    int current_correct_count = neg_distances.size();
    int i = 0;
    int j = 0;
    while (i < pos_distances.size() && j < neg_distances.size())
    {
        if (pos_distances[i] < neg_distances[j])
        {
            current_correct_count++;
            i++;
        } else
        {
            current_correct_count--;
            j++;
        }
        if (current_correct_count > best_correct_count)
        {
            best_correct_count = current_correct_count;
            if (threshold)
                *threshold = pos_distances[i-1];
        }
    }
    best_correct_count = std::max(best_correct_count, (int) pos_distances.size());
    return (float) best_correct_count / (pos_distances.size() + neg_distances.size());
}

float get_triplet_prob(float match_distance, float nonmatch_distance)
{
    return 1.0 / (1.0 + exp(match_distance - nonmatch_distance));
}

float get_triplet_dbl_log_loss(float match_distance, float nonmatch_distance)
{
    float p = get_triplet_prob(match_distance, nonmatch_distance);
    float l = -log(p);
    if (l > 10e10) // overflow? approximate
        l = match_distance - nonmatch_distance;
    return l;
}

template <typename Dtype>
void apply_triplet_dbl_grad(float p,
                            const Dtype* anchor_feature, Dtype* anchor_grad,
                            const Dtype* match_feature, Dtype* match_grad,
                            const Dtype* nonmatch_feature, Dtype* nonmatch_grad,
                            int channels)
{
    for (int k = 0; k < channels; k++)
    {
        anchor_grad[k] += (anchor_feature[k] - match_feature[k] - (anchor_feature[k] - nonmatch_feature[k])) * (1 - p);
        match_grad[k] += (match_feature[k] - anchor_feature[k]) * (1 - p);
        nonmatch_grad[k] += -(nonmatch_feature[k] - anchor_feature[k]) * (1 - p);
    }
}

float get_pair_prob(float d, float m)
{
    CHECK_GT(d, -1e-10);
    return (1 + exp(-m)) / (1 + exp(d - m));
}

float get_pair_dbl_log_loss(float d,  int label, float m)
{
    double p = get_pair_prob(d, m);
    if (label)
        return -log(std::max(p, 1e-10));
    else
        return -log(std::max(1 - p, 1e-10));
}

template <typename Dtype>
void apply_pair_dbl_grad(float d, float m, int label,
                         const Dtype* a_feature, Dtype* a_grad,
                         const Dtype* b_feature, Dtype* b_grad, int channels, float factor = 1)
{
    double f = 0;
    if (label)
    {
        f = exp(d - m) / (1 + exp(d - m)) * 2;
        if (d - m > 111)
            f = 2;
    } else
    {
        f = -exp(d) * (1 + exp(-m)) / (exp(d) - 1) / (1 + exp(d - m)) * 2;
        if (d - m > 111)
            f = 0;
    }

    f = std::max(-2.0, std::min(f, 2.0)) * factor; // for stability

    for (int k = 0; k < channels; k++)
    {
        a_grad[k] += f * (a_feature[k] - b_feature[k]);
        b_grad[k] += f * (b_feature[k] - a_feature[k]);

        CHECK(a_grad[k] >= 0 || a_grad[k] <= 0);
        CHECK(b_grad[k] >= 0 || b_grad[k] <= 0);
    }
}
