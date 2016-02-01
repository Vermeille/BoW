#pragma once

#include <vector>
#include <string>
#include <iostream>

#include <boost/bimap.hpp>

#include "document.h"

class SequenceTagger {
    std::vector<std::vector<double>> word_weight_;
    std::vector<std::vector<double>> state_transition_;

    size_t input_size_;
    size_t output_size_;

    size_t start_word_;
    size_t start_label_;

    size_t stop_word_;
    size_t stop_label_;

    void Init();

    double WordF(Label target, const WordFeatures& w) const;

    void WordF_Backprop(
            const WordFeatures& w, Label truth, const double* probabilities);

    double RunAllFeatures(
            Label k, const WordFeatures& ws, const WordFeatures& prev) const;

    void Backprop(
            const WordFeatures& ws,
            const WordFeatures& prev,
            Label truth,
            const double* probabilities);

  public:
    SequenceTagger(
            size_t in_sz, size_t out_sz,
            size_t start_word, size_t start_label,
            size_t stop_word, size_t stop_label);
    SequenceTagger();

    const std::vector<std::vector<double>>& weights() const { return word_weight_; }
    const std::vector<double>& weights(size_t label) const { return word_weight_[label]; }
    double weight(size_t label, size_t w) const { return word_weight_[label][w]; }

    std::string Serialize() const;

    static SequenceTagger FromSerialized(std::istream& file);

    double ComputeNLL(double* probas) const;

    Label ComputeTagForWord(
            const WordFeatures& ws,
            const WordFeatures& prev,
            double* probabilities) const;

    void Compute(std::vector<WordFeatures>& ws);

    int Train(const Document& doc);

    void ResizeInput(size_t in);
    void ResizeOutput(size_t out);
};

