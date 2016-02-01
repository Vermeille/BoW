#include <glog/logging.h>

#include "sequence-tagger.h"

static const unsigned int kNotFound = -1;
static constexpr double kLearningRate = 0.01;

SequenceTagger::SequenceTagger(
            size_t in_sz, size_t out_sz,
            size_t start_word, size_t start_label,
            size_t stop_word, size_t stop_label)
        : input_size_(in_sz),
        output_size_(out_sz),
        start_word_(start_word),
        start_label_(start_label),
        stop_word_(stop_word),
        stop_label_(stop_label) {
}

SequenceTagger::SequenceTagger()
        : input_size_(0),
        output_size_(0) {
}

void SequenceTagger::Init() {
    word_weight_.resize(output_size_);
    for (size_t i = 0; i < output_size_; ++i) {
        word_weight_[i].resize(input_size_);
        for (size_t j = 0; j < input_size_; ++j) {
            word_weight_[i][j] = 0;
        }
    }
    state_transition_.resize(output_size_);
    for (size_t i = 0; i < output_size_; ++i) {
        state_transition_[i].resize(output_size_);
        for (size_t j = 0; j < output_size_; ++j) {
            state_transition_[i][j] = 0;
        }
    }
}

double SequenceTagger::WordF(Label target, const WordFeatures& w) const {
    if (w.idx == kNotFound)
        return 0;

    return word_weight_[target][w.idx];
}

void SequenceTagger::WordF_Backprop(
        const WordFeatures& w,
        Label truth,
        const double* probabilities) {
    if (w.idx == kNotFound)
        return;

    for (size_t k = 0; k < output_size_; ++k) {
        double target = (truth == k) ? 1 : 0;
        word_weight_[k][w.idx] += kLearningRate * (target - probabilities[k]);
    }
}

double SequenceTagger::RunAllFeatures(
        Label k,
        const WordFeatures& w,
        const WordFeatures& /* prev */) const {
    double sum = 0;
    sum += WordF(k, w);
    return sum;
}

double SequenceTagger::ComputeNLL(double* probas) const {
    double nll = 0;
    for (size_t i = 0; i < output_size_; ++i) {
        nll += std::log(probas[i]);
    }
    return -nll;
}

Label SequenceTagger::ComputeTagForWord(
        const WordFeatures& w,
        const WordFeatures& prev,
        double* probabilities) const {
    double total = 0;
    for (size_t k = 0; k < output_size_; ++k) {
        probabilities[k] = std::exp(RunAllFeatures(k, w, prev));
        total += probabilities[k];
    }

    int max = 0;
    for (size_t k = 0; k < output_size_; ++k) {
        probabilities[k] /= total;
        if (probabilities[k] > probabilities[max]) {
            max = k;
        }
    }
    return max;
}

// TODO: Implement Viterbi!
void SequenceTagger::Compute(std::vector<WordFeatures>& ws) {
    WordFeatures prev("*");
    prev.idx = start_word_;
    prev.pos = start_label_;

    std::vector<double> probas(output_size_);

    for (auto& w : ws) {
        w.pos = ComputeTagForWord(w, prev, probas.data());

        prev = w;
    }
    // FIXME: Use STOP for Viterbi
}

void SequenceTagger::Backprop(
        const WordFeatures& w,
        const WordFeatures& /* prev */,
        Label truth,
        const double* probabilities) {
    WordF_Backprop(w, truth, probabilities);
}

int SequenceTagger::Train(const Document& doc) {
    double nll = 0;
    std::vector<double> probas(output_size_);
    int nb_correct = 0;
    int nb_tokens = 0;

    word_weight_.resize(output_size_);
    for (size_t i = 0; i < output_size_; ++i) {
        word_weight_[i].resize(input_size_);
    }

    state_transition_.resize(output_size_);
    for (size_t i = 0; i < output_size_; ++i) {
        state_transition_[i].resize(output_size_);
    }

    std::cout << "dims: " << input_size_ << " " << output_size_ << std::endl;

    for (auto& ex : doc.examples) {
        WordFeatures prev("*");
        prev.idx = start_word_;
        prev.pos = start_label_;
        for (auto& w : ex.inputs) {
            Label predicted = ComputeTagForWord(w, prev, probas.data());
            nb_correct += predicted == w.pos ? 1 : 0;
            ++nb_tokens;

            nll += ComputeNLL(probas.data());

            Backprop(w, prev, w.pos, probas.data());
            prev = w;
        }
        WordFeatures end("_END_");
        end.idx = stop_word_;
        end.pos = stop_label_;

        Label predicted = ComputeTagForWord(end, prev, probas.data());
        nb_correct += predicted == stop_label_ ? 1 : 0;
        ++nb_tokens;

        nll += ComputeNLL(probas.data());

        Backprop(end, prev, stop_label_, probas.data());
    }
    return  nb_correct * 100 / nb_tokens;
}

std::string SequenceTagger::Serialize() const {
    std::ostringstream out;
    out << input_size_ << " " << output_size_ << " " <<
        start_word_ << " " << start_label_ << " " <<
        stop_word_ << " " << stop_label_ << std::endl;

    for (size_t w = 0; w < input_size_; ++w) {
        for (size_t i = 0; i < output_size_; ++i) {
            out << weight(i, w) << " ";
        }
        out << std::endl;
    }
    return out.str();
}

SequenceTagger SequenceTagger::FromSerialized(std::istream& in) {
    std::string tok;
    size_t in_sz = 0;
    size_t out_sz = 0;
    size_t start_word, start_label, stop_word, stop_label;
    in >> in_sz >> out_sz >>
        start_word >> start_label >>
        stop_word >> stop_label;

    SequenceTagger bow(in_sz, out_sz,
            start_word, start_label,
            stop_word, stop_label);

    for (size_t i = 0; i < bow.output_size_; ++i) {
        bow.word_weight_.emplace_back();
    }

    for (size_t w = 0; w < bow.input_size_; ++w) {
        for (size_t i = 0; i < bow.output_size_; ++i) {
            double score;
            in >> score;
            bow.word_weight_[i].push_back(score);
        }
    }

    return bow;
}

void SequenceTagger::ResizeInput(size_t in) {
    if (in <= input_size_) {
        return;
    }

    for (auto& weights : word_weight_) {
        weights.resize(in);
    }
    input_size_ = in;
}

void SequenceTagger::ResizeOutput(size_t out) {
    if (out <= output_size_) {
        return;
    }

    word_weight_.resize(out);
    for (size_t i = output_size_; i < out; ++i) {
        word_weight_[i].resize(input_size_);
    }

    state_transition_.resize(out);
    for (size_t i = 0; i < out; ++i) {
        state_transition_[i].resize(out);
    }

    output_size_ = out;
}

