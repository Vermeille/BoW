#pragma once

#include <vector>
#include <string>
#include <sstream>

#include <nlp-common/bow.h>
#include <nlp-common/dict.h>
#include <nlp-common/document.h>

struct BowResult {
    std::vector<double> confidence;
    Label label;
    std::vector<WordFeatures> words;
};

class BoWClassifier {
  public:
    size_t Train(const Document& doc);
    BowResult ComputeClass(const std::string& ws);

    Document Parse(const std::string& str);

    LabelSet& labels() { return ls_; }

    const std::vector<std::vector<double>>& weights() const { return bow_.weights(); }
    const std::vector<double>& weights(size_t label) const { return bow_.weights(label); }
    double weight(size_t label, size_t w) const { return bow_.weight(label, w); }

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

