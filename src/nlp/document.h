#pragma once

#include <sstream>
#include <vector>
#include <string>
#include <boost/bimap.hpp>

typedef unsigned int Label;

class LabelSet {
  public:
    Label GetLabel(const std::string& str) {
        return labels_.insert(decltype (labels_)::value_type(str, labels_.size())).first->right;
    }

    void AddLabel(const std::string& str) { GetLabel(str); }

    std::string GetString(Label pos) const {
        return labels_.right.at(pos);
    }

    size_t size() const { return labels_.size(); }

    std::string Serialize() const {
        std::ostringstream out;
        out << labels_.size() << std::endl;

        for (auto& l : labels_.right) {
            out << l.first << " " << l.second << std::endl;
        }

        return out.str();
    }

    static LabelSet FromSerialized(std::istream& in) {
        LabelSet ls;
        size_t nb_labels;
        std::string label;

        in >> nb_labels;
        for (size_t i = 0; i < nb_labels; ++i) {
            Label id;
            in >> id >> label;
            ls.labels_.insert(decltype (labels_)::value_type(label, id));
        }

        return ls;
    }

  private:
    boost::bimap<std::string, Label> labels_;
};

typedef unsigned int Label;

struct WordFeatures {
    std::string str;
    size_t idx;

    size_t pos;

    WordFeatures(const std::string& s) : str(s), idx(0) {}
};

struct TrainingExample {
    std::vector<WordFeatures> inputs;
    Label output;
};

struct Document {
    std::vector<TrainingExample> examples;
};

