#include <glog/logging.h>

#include "bow.h"

static const unsigned int kNotFound = -1;
static const double kLearningRate = 0.001;

static double randr(float from, float to) {
    double distance = to - from;
    return ((double)rand() / ((double)RAND_MAX + 1) * distance) + from;
}

BagOfWords::BagOfWords(size_t in_sz, size_t out_sz)
        : w_weights_(std::make_shared<Eigen::MatrixXd>(out_sz, in_sz)),
        b_weights_(std::make_shared<Eigen::MatrixXd>(out_sz, 1)),
        input_size_(in_sz),
        output_size_(out_sz) {
    auto& b_mat = *b_weights_;
    auto& w_mat = *w_weights_;

    for (size_t i = 0; i < out_sz; ++i) {
        b_mat(i, 0) = randr(-1, 1);

        for (size_t j = 0; j < in_sz; ++j) {
            w_mat(i, j) = randr(-1, 1);
        }
    }
}

BagOfWords::BagOfWords()
        : w_weights_(std::make_shared<Eigen::MatrixXd>(0, 0)),
        b_weights_(std::make_shared<Eigen::MatrixXd>(0, 1)),
        input_size_(0),
        output_size_(0) {
}

ad::Var BagOfWords::ComputeModel(
        ad::ComputationGraph& g, ad::Var& w, ad::Var& b,
        const std::vector<WordFeatures>& ws) const {
    Eigen::MatrixXd input(input_size_, 1);
    input.setZero();

    for (auto& wf : ws) {
        // one hot encode each word
        if (wf.idx < input_size_) {
            input(wf.idx, 0) = 1;
        }
    }

    ad::Var x = g.CreateParam(input);

    return ad::Softmax(w * x + b);
}

Eigen::MatrixXd BagOfWords::ComputeClass(const std::vector<WordFeatures>& ws) const {
    ad::ComputationGraph g;
    ad::Var w = g.CreateParam(w_weights_);
    ad::Var b = g.CreateParam(b_weights_);

    return ComputeModel(g, w, b, ws).value();
}

int BagOfWords::Train(const Document& doc) {
    double nll = 0;
    int nb_correct = 0;
    int nb_tokens = 0;

    for (auto& ex : doc.examples) {
        using namespace ad;

        Eigen::MatrixXd y_mat(output_size_, 1);
        y_mat.setZero();
        y_mat(ex.output, 0) = 1;

        ComputationGraph g;
        Var w = g.CreateParam(w_weights_);
        Var b = g.CreateParam(b_weights_);
        Var y = g.CreateParam(y_mat);

        Var h = ComputeModel(g, w, b, ex.inputs);

        // MSE is weirdly doing better than Cross Entropy
        Var J = ad::MSE(y, h);

        opt::SGD sgd(0.01);
        g.BackpropFrom(J);
        g.Update(sgd, {&w, &b});

        Eigen::MatrixXd& h_mat = h.value();
        Eigen::MatrixXd::Index max_row, max_col;
        h_mat.maxCoeff(&max_row, &max_col);
        Label predicted = max_row;
        nb_correct += predicted == ex.output ? 1 : 0;
        ++nb_tokens;

        nll += J.value()(0, 0);
    }
    return  nb_correct * 100 / nb_tokens;
}

std::string BagOfWords::Serialize() const {
    std::ostringstream out;

    out << input_size_ << " " << output_size_ << std::endl;

    auto& b_mat = *b_weights_;
    auto& w_mat = *w_weights_;

    for (size_t w = 0; w < output_size_; ++w) {
        for (size_t i = 0; i < input_size_; ++i) {
            out << w_mat(w, i) << " ";
        }
        out << std::endl;
    }

    for (size_t w = 0; w < output_size_; ++w) {
        out << b_mat(w, 0) << "\n";
    }

    return out.str();
}

BagOfWords BagOfWords::FromSerialized(std::istream& in) {
    std::string tok;
    size_t in_sz = 0;
    size_t out_sz = 0;
    in >> in_sz >> out_sz;

    BagOfWords bow(in_sz, out_sz);
    auto& b_mat = *bow.b_weights_;
    auto& w_mat = *bow.w_weights_;

    for (size_t w = 0; w < bow.output_size_; ++w) {
        for (size_t i = 0; i < bow.input_size_; ++i) {
            double score;
            in >> score;
            w_mat(w, i) = score;
        }
    }

    for (size_t w = 0; w < bow.output_size_; ++w) {
        double score;
        in >> score;
        b_mat(w, 0) = score;
    }

    return bow;
}

void BagOfWords::ResizeInput(size_t in) {
    if (in <= input_size_) {
        return;
    }

    Eigen::MatrixXd& w_mat = *w_weights_;
    w_mat.conservativeResize(output_size_, in);

    for (int row = 0, nb_rows = w_weights_->rows(); row < nb_rows; ++row) {
        for (size_t i = input_size_; i < in; ++i) {
            w_mat(row, i) = randr(-1, 1);
        }
    }
    input_size_ = in;
}

void BagOfWords::ResizeOutput(size_t out) {
    if (out <= output_size_) {
        return;
    }

    Eigen::MatrixXd& w_mat = *w_weights_;
    w_mat.conservativeResize(out, input_size_);
    Eigen::MatrixXd& b_mat = *b_weights_;
    b_mat.conservativeResize(out, 1);

    for (unsigned row = output_size_; row < out; ++row) {
        for (unsigned  col = 0, nb_cols = w_weights_->cols(); col < nb_cols; ++col) {
            w_mat(row, col) = randr(-1, 1);
        }
    }

    for (unsigned row = output_size_; row < out; ++row) {
        b_mat(row, 0) = randr(-1, 1);
    }

    output_size_ = out;
}

