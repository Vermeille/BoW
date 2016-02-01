#pragma once

#include <vector>
#include <string>
#include <iostream>

#include <Eigen/Dense>
#include <ad/ad.h>

#include "document.h"

class BagOfWords {
    std::shared_ptr<Eigen::MatrixXd> w_weights_; //(out_sz, in_sz)
    std::shared_ptr<Eigen::MatrixXd> b_weights_; //(out_sz, 1)
    size_t input_size_;
    size_t output_size_;

    ad::Var ComputeModel(
            ad::ComputationGraph& g, ad::Var& w, ad::Var& b,
            const std::vector<WordFeatures>& ws) const;

  public:
    BagOfWords(size_t in_sz, size_t out_sz);
    BagOfWords();

    std::string Serialize() const;
    static BagOfWords FromSerialized(std::istream& file);

    Eigen::MatrixXd ComputeClass(const std::vector<WordFeatures>& ws) const;

    int Train(const Document& doc);

    void ResizeInput(size_t in);
    void ResizeOutput(size_t out);
};

