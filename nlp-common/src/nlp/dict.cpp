#include "dict.h"

#include <sstream>

Dictionnary::Dictionnary()
        : max_freq_(0) {
    unk_id_ = GetWordId("_UNK_");
}

size_t Dictionnary::GetWordId(const std::string& w) {
    size_t id = dict_.insert(decltype (dict_)::value_type(w, dict_.size())).first->right;
    if (stats_.size() < id + 1) {
        stats_.resize(id + 1);
        stats_[id] = 1;
    } else {
        ++stats_[id];
    }
    return id;
}

size_t Dictionnary::GetWordIdOrUnk(const std::string& w) {
    auto res = dict_.left.find(w);
    if (res == dict_.left.end()) {
        return unk_id_;
    } else {
        ++stats_[res->second];
        max_freq_ = std::max(max_freq_, stats_[res->second]);
        return res->second;
    }
}

void NGramMaker::Annotate(std::vector<WordFeatures>& sentence) {
    for (auto& s : sentence) {
        s.idx = dict_.GetWordIdOrUnk(s.str);
    }
}

void NGramMaker::Learn(std::vector<WordFeatures>& sentence) {
    for (auto& s : sentence) {
        s.idx = dict_.GetWordId(s.str);
    }
}

std::string Dictionnary::Serialize() const {
    std::ostringstream out;
    out << dict_.size() << std::endl;
    for (auto& w : dict_.left) {
        out << w.first << " " << w.second << " " << stats_[w.second] << std::endl;
    }
    return out.str();
}

Dictionnary Dictionnary::FromSerialized(std::istream& in) {
    Dictionnary dict;
    size_t nb;
    in >> nb;

    std::string w;
    size_t id;
    dict.stats_.resize(nb);
    for (size_t i = 0; i < nb; ++i) {
        in >> w >> id;
        in >> dict.stats_[id];
        dict.dict_.insert(decltype (dict.dict_)::value_type(w, id));
    }

    return dict;
}
