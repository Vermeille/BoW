#pragma once

#include <vector>
#include <string>

#include <boost/bimap.hpp>

#include "document.h"

class BagOfWords {
    std::vector<std::vector<double>> word_weight_;
    boost::bimap<std::string, int> dict_;
    LabelSet labels_;

    void ZeroInit();

    double WordF(Label target, unsigned int w) const;
    void WordF_Backprop(unsigned int w, Label truth, const double* probabilities);
    double RunAllFeatures(Label k, const std::vector<unsigned int>& ws) const;

  public:
    LabelSet& labels() { return labels_; }
    const std::vector<std::vector<double>>& weights() const { return word_weight_; }
    const std::vector<double>& weights(size_t label) const { return word_weight_[label]; }
    double weight(size_t label, size_t w) const { return word_weight_[label][w]; }

    std::string Serialize() const;

    double ComputeNLL(double* probas) const;
    size_t GetWordId(const std::string& w);
    size_t GetWordIdOrUnk(const std::string& w) const;
    size_t GetVocabSize() const { return dict_.size(); }

    std::string WordFromId(size_t id) const;

    static BagOfWords FromSerialized(const std::string& file);

    bool IsInVocab(const std::string& w) { return dict_.left.find(w) != dict_.left.end(); }
    bool Init();
    Label ComputeClass(const std::vector<unsigned int>& ws, double* probabilities) const;
    void Backprop(const std::vector<unsigned int>& ws, Label truth, const double* probabilities);
    int Train(const Document& doc);
};

