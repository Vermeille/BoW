#include <vector>
#include <string>
#include <utility>
#include <map>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <fstream>

#include "bow.h"
#include "document.h"

#include <httpi/job.h>
#include <httpi/displayer.h>

static const unsigned int kNotFound = -1;

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
        while (input) {
            ws.push_back(g_bow.GetWordIdOrUnk(w));
            input >> w;
        }

        std::vector<double> probas(g_bow.labels().size());

        Label k = g_bow.ComputeClass(ws, probas.data());

        // Header
        Html html;
        for (auto w : ws) {
            if (w != kNotFound) {
                html <<
                    Tag("span").Attr("style",
                        "font-size: " + std::to_string((1 + std::log(1 + std::abs(g_bow.weights()[k][w]))) * 30) + "px;"
                        "color: " + std::string(g_bow.weights()[k][w] > 0 ? "green" : "red") + ";")
                        << g_bow.WordFromId(w) << " " << Close();
            } else {
                html << Tag("span") << "_UNK_ " << Close();
            }
        }
        html <<
            P() << "best prediction: " << g_bow.labels().GetString(k) << " " << std::to_string(probas[k] * 100)
                << Close() <<

        // table of global confidence
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
