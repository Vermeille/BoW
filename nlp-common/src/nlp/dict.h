#pragma once

#include <string>
#include <vector>

#include <boost/bimap.hpp>

#include "featurizer.h"

class Dictionnary {
    boost::bimap<std::string, size_t> dict_;
    mutable std::vector<size_t> stats_;
    size_t max_freq_;
    size_t unk_id_;

  public:
    Dictionnary();
    bool IsInVocab(const std::string& w) { return dict_.left.find(w) != dict_.left.end(); }
    size_t GetWordId(const std::string& w);
    size_t GetWordIdOrUnk(const std::string& w);
    size_t size() const { return dict_.size(); }
    std::string Serialize() const;
    static Dictionnary FromSerialized(std::istream& in);
    boost::bimap<std::string, size_t>::left_map::iterator begin() { return dict_.left.begin(); }
    boost::bimap<std::string, size_t>::left_map::iterator end() { return dict_.left.end(); }
    std::string WordFromId(size_t id) const { return dict_.right.at(id); }
};

class NGramMaker {
    Dictionnary dict_;
  public:
    void Annotate(std::vector<WordFeatures>& sentence);
    void Learn(std::vector<WordFeatures>& sentence);
    const Dictionnary& dict() const { return dict_; }
    std::string WordFromId(size_t id) const { return dict_.WordFromId(id); }
    std::string Serialize() const { return dict_.Serialize(); }
    static NGramMaker FromSerialized(std::istream& in) {
        NGramMaker ngram;
        ngram.dict_ = Dictionnary::FromSerialized(in);
        return ngram;
    }
};
