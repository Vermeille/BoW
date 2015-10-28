#include <vector>
#include <fstream>
#include <string>
#include <utility>
#include <map>
#include <cmath>
#include <iostream>
#include <algorithm>

#include <boost/bimap.hpp>

#include <httpi/job.h>
#include <httpi/displayer.h>

static const int kVocabSize = 100;
static const double kLearningRate = 0.001;
static const unsigned int kNotFound = -1;

typedef unsigned int Label;
boost::bimap<std::string, Label> labels;

Label TextToPOS(const std::string& str) {
    return labels.insert(decltype (labels)::value_type(str, labels.size())).first->right;
};

std::string POSToText(Label pos) {
    return labels.right.at(pos);
};

typedef std::vector<std::pair<std::vector<unsigned int>, Label> > Document;

struct WordFeatures {
    std::string as_string;
    unsigned int idx;

    WordFeatures() {}
    WordFeatures(const std::string& str) : as_string(str) {}
};

inline bool ends_with(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size())
        return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

inline bool starts_with(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size())
        return false;
    return std::equal(ending.begin(), ending.end(), value.begin());
}
/* WORD WEIGHT */

std::vector<std::vector<double>> word_weight;

double WordF(Label target, unsigned int w) {
    if (w == kNotFound)
        return 0;

    return word_weight[target][w];
}

void WordF_Backprop(unsigned int w, Label truth, const double* probabilities) {
    if (w == kNotFound)
        return;

    for (int k = 0; k < labels.size(); ++k) {
        double target = (truth == k) ? 1 : 0;
        word_weight[k][w] += kLearningRate * (target - probabilities[k]);
    }
}

double RunAllFeatures(Label k, const std::vector<unsigned int>& ws) {
    double sum = 0;
    for (int i = 0; i < ws.size(); ++i) {
        sum += WordF(k, ws[i]);
    }
    return sum;
}

Label ComputeClass(const std::vector<unsigned int>& ws, double* probabilities) {
    double total = 0;
    for (int k = 0; k < labels.size(); ++k) {
        probabilities[k] = std::exp(RunAllFeatures(k, ws));
        total += probabilities[k];
    }

    int max = 0;
    for (int k = 0; k < labels.size(); ++k) {
        probabilities[k] /= total;
        if (probabilities[k] > probabilities[max]) {
            max = k;
        }
    }
    return max;
}

void Backprop(const std::vector<unsigned int>& ws, Label truth, const double* probabilities) {
    for (int i = 0; i < ws.size(); ++i) {
        WordF_Backprop(ws[i], truth, probabilities);
    }
}

double ComputeNLL(double* probas) {
    double nll = 0;
    for (int i = 0; i < labels.size(); ++i) {
        nll += std::log(probas[i]);
    }
    return -nll;
}

std::map<std::string, int> dict;

void ZeroInit() {
    for (int i = 0; i < labels.size(); ++i) {
        word_weight[i].resize(kVocabSize);
        for (int j = 0; j < kVocabSize; ++j) {
            word_weight[i][j] = 0;
        }
    }
}

void Save() {
    std::ofstream out("params.bow");

    out << dict.size() << "\n";
    for (auto& w : dict) {
        out << w.first << " " << w.second << "\n";
    }

    for (int i = 0; i < labels.size(); ++i) {
        word_weight[i].resize(kVocabSize);
        for (int j = 0; j < kVocabSize; ++j) {
            out << word_weight[i][j] << " ";
        }
        out << "\n";
    }
}

bool Init() {
    std::ifstream in("params.bow");

    if (!in) {
        ZeroInit();
        return true;
    }

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
    return false;
}

Document BuildDocument(char* filename) {
    Document doc;

    std::ifstream input(filename);
    std::string w;
    std::string pos;
    unsigned int max_word_id = 0;

    doc.push_back(std::make_pair(std::vector<unsigned int>(), 0));
    while (input) {
        unsigned word_id = max_word_id;
        input >> w;

        if (w == "|") {
            input >> w;
            doc.back().second = TextToPOS(w);
            doc.push_back(std::make_pair(std::vector<unsigned int>(), 0));
        } else {
            auto res = dict.insert(std::make_pair(w, max_word_id));
            if (!res.second) {
                word_id = res.first->second;
            } else {
                ++max_word_id;
            }

            doc.back().first.push_back(word_id);
        }
    }
    word_weight.resize(labels.size());
    return doc;
}

void Train(const Document& doc) {
    for (int epoch = 0; epoch < 50; ++epoch) {
        double nll = 0;
        double probas[labels.size()];
        int nb_correct = 0;
        int nb_tokens = 0;
        for (size_t i = 0; i < doc.size(); ++i) {
            Label predicted = ComputeClass(doc[i].first, probas);
            nb_correct += predicted == doc[i].second ? 1 : 0;
            ++nb_tokens;

            nll += ComputeNLL(probas);

            Backprop(doc[i].first, doc[i].second, probas);

            if (i % 1000 == 0) {
                std::cout << nb_correct << " / " << nb_tokens << " (" << ((double) nb_correct *100 / nb_tokens) << "%)" << std::endl;
            }

            if (predicted != doc[i].second) {
                for (auto& w : doc[i].first)
                    std::cout << w << " ";
                std::cout << POSToText(predicted) << " instead of " << POSToText(doc[i].second) << "\n";
            }
        }
        std::cout << nll << "\n" << nb_correct << " / " << nb_tokens << "\n=======\n";
    }
}

static const JobDesc classify = {
    { { "input", "text", "Text to classify" } },
    "Classify",  // name
    "/classify",  // url
    "Classify the input text to one of the categories",  // longer description
    true /* synchronous */,
    true /* reentrant */,
    [](const std::vector<std::string>& vs, size_t) { // the actual function
        std::istringstream input(vs[0]);
        std::string w;
        std::vector<unsigned int> ws;
        input >> w;
        while (w != "." && input) {
            auto res = dict.find(w);
            ws.push_back(res == dict.end() ? kNotFound : res->second);
            std::cout << "wid: " << ws.back() << "\n";
            input >> w;
        }

        std::vector<double> probas(labels.size());

        Label k = ComputeClass(ws, probas.data());

        Html html;
        html << P() <<"input: " << vs[0] << Close() <<
            P() << "best prediction: " << POSToText(k) << " " << std::to_string(probas[k] * 100)
                << Close() <<

            Tag("table").AddClass("table") <<
                Tag("tr") <<
                    Tag("th") << "Label" << Close() <<
                    Tag("th") << "Confidence" << Close() <<
                Close();

        for (int i = 0; i < labels.size(); ++i) {
            html <<
                Tag("tr") <<
                    Tag("td") << POSToText(i) << Close() <<
                    Tag("td") << std::to_string(probas[i]) << Close() <<
                Close();
        }
        html << Close();
        return html;
    }
};

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: ./" << argv[0] << " <training set>\n";
        return 1;
    }
    Init();
    Document doc = BuildDocument(argv[1]);

    bool need_training = Init();

    if (need_training) {
        Train(doc);
        Save();
    }

    InitHttpInterface();  // Init the http server
    RegisterJob(classify);
    ServiceLoopForever();  // infinite loop ending only on SIGINT / SIGTERM / SIGKILL
    StopHttpInterface();  // clear resources
    return 0;
}
