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

class BagOfWords {
    std::vector<std::vector<double>> word_weight_;
    boost::bimap<std::string, int> dict_;
    LabelSet labels_;

    void ZeroInit() {
        word_weight_.resize(labels_.size());
        for (int i = 0; i < labels_.size(); ++i) {
            word_weight_[i].resize(kVocabSize);
            for (int j = 0; j < kVocabSize; ++j) {
                word_weight_[i][j] = 0;
            }
        }
    }

    double WordF(Label target, unsigned int w) {
        if (w == kNotFound)
            return 0;

        return word_weight_[target][w];
    }

    void WordF_Backprop(unsigned int w, Label truth, const double* probabilities) {
        if (w == kNotFound)
            return;

        for (int k = 0; k < labels_.size(); ++k) {
            double target = (truth == k) ? 1 : 0;
            word_weight_[k][w] += kLearningRate * (target - probabilities[k]);
        }
    }

    double RunAllFeatures(Label k, const std::vector<unsigned int>& ws) {
        double sum = 0;
        for (int i = 0; i < ws.size(); ++i) {
            sum += WordF(k, ws[i]);
        }
        return sum;
    }

    public:
    LabelSet& labels() { return labels_; }
    const std::vector<std::vector<double>>& weights() const { return word_weight_; }

    double ComputeNLL(double* probas) {
        double nll = 0;
        for (int i = 0; i < labels_.size(); ++i) {
            nll += std::log(probas[i]);
        }
        return -nll;
    }

    size_t GetWordId(const std::string& w) {
        return dict_.insert(decltype (dict_)::value_type(w, dict_.size())).first->right;
    }

    size_t GetWordIdOrUnk(const std::string& w) const {
        auto res = dict_.left.find(w);
        if (res == dict_.left.end()) {
            return kNotFound;
        } else {
            return res->second;
        }
    }

    size_t GetVocabSize() const { return dict_.size(); }

    const std::string& WordFromId(size_t id) {
        return dict_.right.at(id);
    }

    bool Init() {
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

    Label ComputeClass(const std::vector<unsigned int>& ws, double* probabilities) {
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

    void Backprop(const std::vector<unsigned int>& ws, Label truth, const double* probabilities) {
        for (int i = 0; i < ws.size(); ++i) {
            WordF_Backprop(ws[i], truth, probabilities);
        }
    }

    int Train(const Document& doc) {
        double nll = 0;
        double probas[labels_.size()];
        int nb_correct = 0;
        int nb_tokens = 0;

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
};

BagOfWords g_bow;

/*
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
*/

Document BuildDocument(std::ifstream& input, BagOfWords& bow) {
    Document doc;

    std::string w;
    std::string pos;

    TrainingExample ex;
    input >> w;
    while (input) {

        if (w == "|") {
            input >> w;
            ex.output = bow.labels().GetLabel(w);
            doc.examples.push_back(ex);
            ex = TrainingExample();
        } else {
            ex.inputs.push_back(bow.GetWordId(w));
        }
        input >> w;
    }
    return doc;
}

static const JobDesc classify = {
    { { "input", "text", "Text to classify" } },
    "Classify",  // name
    "/classify",  // url
    "Classify the input text to one of the categories",  // longer description
    true /* synchronous */,
    true /* reentrant */,
    [](const std::vector<std::string>& vs, JobResult& job) { // the actual function
        std::istringstream input(vs[0]);
        std::string w;
        std::vector<unsigned int> ws;
        input >> w;
        while (w != "." && input) {
            ws.push_back(g_bow.GetWordIdOrUnk(w));
            input >> w;
        }

        std::vector<double> probas(g_bow.labels().size());

        Label k = g_bow.ComputeClass(ws, probas.data());

        Html html;
        html << P() <<"input: " << vs[0] << Close() <<
            P() << "best prediction: " << g_bow.labels().GetString(k) << " " << std::to_string(probas[k] * 100)
                << Close() <<

            Tag("table").AddClass("table") <<
                Tag("tr") <<
                    Tag("th") << "Label" << Close() <<
                    Tag("th") << "Confidence" << Close() <<
                Close();

        for (int i = 0; i < g_bow.labels().size(); ++i) {
            html <<
                Tag("tr") <<
                    Tag("td") << g_bow.labels().GetString(i) << Close() <<
                    Tag("td") <<
                        Div().Attr("style",
                                "width: " + std::to_string(200 * probas[i]) + "px;"
                                "padding-left: 2px;"
                                "background-color: lightgreen;") <<
                            std::to_string(probas[i] * 100) <<
                        Close() <<
                    Close() <<
                Close();
        }
        html << Close();
        job.SetPage(html);
    },
};

static const JobDesc train = {
    { { "training set", "text", "A relative or absolute filepath to the training set" } },
    "Train",
    "/train",
    "Train the model on the specified dataset",
    false /* synchronous */,
    false /* reentrant */,
    [](const std::vector<std::string>& vs, JobResult& job) {
        Chart accuracy_chart("accuracy");
        accuracy_chart.Label("iter").Value("accuracy");

        std::ifstream input(vs[0].c_str());
        if (!input) {
            return Html() << "Error: training set file not found";
        }

        Document doc = BuildDocument(input, g_bow);
        input.close();
        g_bow.Init();

        for (int epoch = 0; epoch < 10; ++epoch) {
            int accuracy = g_bow.Train(doc);

            accuracy_chart.Log("accuracy", accuracy);
            accuracy_chart.Log("iter", epoch);
            job.SetPage(Html() << accuracy_chart.Get());
        }

        return Html();
    },
};

Html DisplayWeights(const std::string&, const POSTValues&) {
    Html html;
    for (int label = 0; label < g_bow.labels().size(); ++label) {
        html << H2() << g_bow.labels().GetString(label) << Close();

        auto minmax = std::minmax_element(g_bow.weights()[label].begin(), g_bow.weights()[label].end());
        double max = std::max(std::abs(*minmax.first), std::abs(*minmax.second));

        html <<
        Tag("table").AddClass("table") <<
            Tag("tr") <<
                Tag("th") << "Word" << Close() <<
                Tag("th") << "Score" << Close() <<
                Tag("th") << "Word" << Close() <<
                Tag("th") << "Score" << Close() <<
            Close();

        int vocab = g_bow.GetVocabSize() / 2;
        for (int w = 0; w < g_bow.GetVocabSize() / 2; ++w) {
            html <<
            Tag("tr") <<
                Tag("td") << g_bow.WordFromId(w) << Close() <<
                Tag("td") <<
                    Div().Attr("style",
                            "width: " + std::to_string(200 * std::abs(g_bow.weights()[label][w] / max)) + "px;"
                            "padding-left: 2px;"
                            "background-color: " +
                            std::string(g_bow.weights()[label][w] > 0 ? "lightgreen;" : "salmon;")) <<
                        std::to_string(g_bow.weights()[label][w] * 100) <<
                    Close() <<
                Close() <<
                Tag("td") << g_bow.WordFromId(w + vocab) << Close() <<
                Tag("td") <<
                    Div().Attr("style",
                            "width: " + std::to_string(200 * std::abs(g_bow.weights()[label][w + vocab] / max)) + "px;"
                            "padding-left: 2px;"
                            "background-color: " +
                            std::string(g_bow.weights()[label][w + vocab] > 0 ? "lightgreen;" : "salmon;")) <<
                        std::to_string(g_bow.weights()[label][w + vocab] * 100) <<
                    Close() <<
                Close() <<
            Close();
        }
        html << Close();
    }
    return html;
}

int main(int argc, char** argv) {
    InitHttpInterface();  // Init the http server
    RegisterJob(classify);
    RegisterJob(train);
    RegisterUrl("/weigth", DisplayWeights);
    ServiceLoopForever();  // infinite loop ending only on SIGINT / SIGTERM / SIGKILL
    StopHttpInterface();  // clear resources
    return 0;
}
