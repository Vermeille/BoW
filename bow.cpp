#include <fstream>
#include <glog/logging.h>

#include "bow.h"

static const size_t kVocabSize = 2;
static const unsigned int kNotFound = -1;
static const double kLearningRate = 0.001;

void BagOfWords::ZeroInit() {
    word_weight_.resize(labels_.size());
    for (int i = 0; i < labels_.size(); ++i) {
        word_weight_[i].resize(kVocabSize);
        for (int j = 0; j < kVocabSize; ++j) {
            word_weight_[i][j] = 0;
        }
    }
}

double BagOfWords::WordF(Label target, unsigned int w) const {
    if (w == kNotFound)
        return 0;

    return word_weight_[target][w];
}

void BagOfWords::WordF_Backprop(unsigned int w, Label truth, const double* probabilities) {
    if (w == kNotFound)
        return;

    for (int k = 0; k < labels_.size(); ++k) {
        double target = (truth == k) ? 1 : 0;
        word_weight_[k][w] += kLearningRate * (target - probabilities[k]);
    }
}

double BagOfWords::RunAllFeatures(Label k, const std::vector<unsigned int>& ws) const {
    double sum = 0;
    for (int i = 0; i < ws.size(); ++i) {
        sum += WordF(k, ws[i]);
    }
    return sum;
}

double BagOfWords::ComputeNLL(double* probas) const {
    double nll = 0;
    for (int i = 0; i < labels_.size(); ++i) {
        nll += std::log(probas[i]);
    }
    return -nll;
}

size_t BagOfWords::GetWordId(const std::string& w) {
    size_t id = dict_.insert(decltype (dict_)::value_type(w, dict_.size())).first->right;
    return id;
}

size_t BagOfWords::GetWordIdOrUnk(const std::string& w) const {
    auto res = dict_.left.find(w);
    if (res == dict_.left.end()) {
        return kNotFound;
    } else {
        return res->second;
    }
}

std::string BagOfWords::WordFromId(size_t id) const {
    auto res = dict_.right.find(id);
    return res == dict_.right.end() ? "<UNK>" : res->second;
}

bool BagOfWords::Init() {
    std::ifstream in("params.bow");

    if (!in) {
        ZeroInit();
        return true;
    }

    /*
       unsigned int size;
       std::string w;
       in >> size;
       for (int i = 0; i < size; ++i) {
       in >> w;
       in >> dict[w];
       }

       for (int i = 0; i < labels.size(); ++i) {
       word_weight[i].resize(kVocabSize);
       for (int j = 0; j < kVocabSize; ++j) {
       in >> word_weight[i][j];
       }
       }
       */
    return false;
}

Label BagOfWords::ComputeClass(const std::vector<unsigned int>& ws, double* probabilities) const {
    double total = 0;
    for (int k = 0; k < labels_.size(); ++k) {
        probabilities[k] = std::exp(RunAllFeatures(k, ws));
        total += probabilities[k];
    }

    int max = 0;
    for (int k = 0; k < labels_.size(); ++k) {
        probabilities[k] /= total;
        if (probabilities[k] > probabilities[max]) {
            max = k;
        }
    }
    return max;
}

void BagOfWords::Backprop(const std::vector<unsigned int>& ws, Label truth, const double* probabilities) {
    for (int i = 0; i < ws.size(); ++i) {
        WordF_Backprop(ws[i], truth, probabilities);
    }
}

int BagOfWords::Train(const Document& doc) {
    double nll = 0;
    double probas[labels_.size()];
    int nb_correct = 0;
    int nb_tokens = 0;

    word_weight_.resize(labels_.size());
    for (int i = 0; i < labels_.size(); ++i) {
        word_weight_[i].resize(dict_.size());
    }

    for (auto& ex : doc.examples) {
        Label predicted = ComputeClass(ex.inputs, probas);
        nb_correct += predicted == ex.output ? 1 : 0;
        ++nb_tokens;

        nll += ComputeNLL(probas);

        Backprop(ex.inputs, ex.output, probas);

        if (predicted != ex.output) {
            std::cout << labels_.GetString(predicted)
                << " instead of " << labels_.GetString(ex.output) << "\n";
        }
    }
    return  nb_correct * 100 / nb_tokens;
}

std::string BagOfWords::Serialize() const {
    std::ostringstream out;
    out << labels_.size() << " " << dict_.size() << std::endl;
    for (int i = 0; i < labels_.size(); ++i) {
        out << labels_.GetString(i) << " ";
    }
    out << std::endl;

    for (int w = 0; w < GetVocabSize(); ++w) {
        out << WordFromId(w) << " ";
        for (int i = 0; i < labels_.size(); ++i) {
            out << weight(i, w) << " ";
        }
        out << std::endl;
    }
    return out.str();
}

BagOfWords BagOfWords::FromSerialized(const std::string& file) {
    BagOfWords bow;
    std::istringstream in(file);

    size_t nb_labels;
    size_t nb_words;
    std::string tok;
    in >> nb_labels >> nb_words;

    for (int i = 0; i < nb_labels; ++i) {
        in >> tok;
        bow.labels_.AddLabel(tok);

        bow.word_weight_.push_back({});
    }

    for (int w = 0; w < nb_words; ++w) {
        in >> tok;
        LOG_IF(FATAL, bow.GetWordId(tok) != w) << "incoherence in word index while deserializing";
        for (int i = 0; i < nb_labels; ++i) {
            double score;
            in >> score;
            bow.word_weight_[i].push_back(score);
        }
    }

    return bow;
}
