#pragma once

#include <vector>
#include <string>
#include <sstream>

#include <nlp/bow.h>
#include <nlp/dict.h>
#include <nlp/document.h>

#include <Eigen/Dense>

struct BowResult {
    Eigen::MatrixXd confidence;
    Label label;
    std::vector<WordFeatures> words;
};

class BoWClassifier {
  public:
    size_t Train(const Document& doc);
    BowResult ComputeClass(const std::string& ws);

    Document Parse(const std::string& str);

    LabelSet& labels() { return ls_; }

    double weights(size_t label, size_t w) const { return bow_.weights(label, w); }
    Eigen::MatrixXd& weights() const { return bow_.weights(); }

    std::string WordFromId(size_t id) const { return ngram_.WordFromId(id); }

    size_t OutputSize() const { return ls_.size(); }
    size_t GetVocabSize() const { return ngram_.dict().size(); }

    static BoWClassifier FromSerialized(std::istream& in);

    std::string Serialize() const {
        return ngram_.Serialize() + bow_.Serialize() + ls_.Serialize();
    }

    BoWClassifier() : bow_(0, 0) {}

  private:
    NGramMaker ngram_;
    BagOfWords bow_;
    LabelSet ls_;
};

