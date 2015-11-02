#pragma once

#include <vector>

typedef unsigned int Label;

class LabelSet {
  public:
    Label GetLabel(const std::string& str) {
        return labels_.insert(decltype (labels_)::value_type(str, labels_.size())).first->right;
    };

    std::string GetString(Label pos) {
        return labels_.right.at(pos);
    };

    size_t size() const { return labels_.size(); }

  private:
    boost::bimap<std::string, Label> labels_;
};

struct TrainingExample {
    std::vector<unsigned int> inputs;
    Label output;
};

struct Document {
    std::vector<TrainingExample> examples;
};

