#include <vector>
#include <string>
#include <utility>
#include <map>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <fstream>

#include "bow.h"

#include <httpi/job.h>
#include <httpi/displayer.h>

static const unsigned int kNotFound = -1;

BoWClassifier g_bow;

static const JobDesc classify = {
    { { "input", "text", "Text to classify" } },
    "Classify",  // name
    "/classify",  // url
    "Classify the input text to one of the categories",  // longer description
    true /* synchronous */,
    true /* reentrant */,
    [](const std::vector<std::string>& vs, JobResult& job) { // the actual function
        std::vector<double> probas(g_bow.OutputSize());
        auto pair = g_bow.ComputeClass(vs[0], probas.data());
        Label k = pair.first;
        auto& ws = pair.second;

        // Header
        Html html;
        for (auto w : ws) {
            if (w.idx != kNotFound) {
                html <<
                    Tag("span").Attr("style",
                        "font-size: " + std::to_string((1 + std::log(1 + std::abs(g_bow.weight(k, w.idx)))) * 30) + "px;"
                        "color: " + std::string(g_bow.weight(k, w.idx) > 0 ? "green" : "red") + ";")
                        << g_bow.WordFromId(w.idx) << " " << Close();
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

        for (size_t i = 0; i < g_bow.labels().size(); ++i) {
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
    { { "trainingset", "file", "A training set" },
      { "epochs", "number", "How many iterations?" } },
    "Train",
    "/train",
    "Train the model on the specified dataset",
    false /* synchronous */,
    false /* reentrant */,
    [](const std::vector<std::string>& vs, JobResult& job) {
        const size_t nb_epoch = std::atoi(vs[1].c_str());
        std::cout << "epochs: " << nb_epoch << "\n";
        Document doc = g_bow.Parse(vs[0]);
        Chart accuracy_chart("accuracy");
        accuracy_chart.Label("iter").Value("accuracy");

        for (size_t epoch = 0; epoch < nb_epoch; ++epoch) {
            int accuracy = g_bow.Train(doc);

            accuracy_chart.Log("accuracy", accuracy);
            accuracy_chart.Log("iter", epoch);
            job.SetPage(Html() << accuracy_chart.Get());
        }
    },
};

static const JobDesc load = {
    { { "model", "file", "The model file" } },
    "Load",
    "/load",
    "Load a model",
    true /* synchronous */,
    false /* reentrant */,
    [](const std::vector<std::string>& vs, JobResult& job) {
        std::istringstream in(vs[0]);
        g_bow = BoWClassifier::FromSerialized(in);
        job.SetPage(Html() << "done");
    },
};

static const JobDesc training_single = {
    { { "input", "text", "An input sentence" },
      { "label", "text", "The label" } },
    "Single example",
    "/training_single",
    "Load a model",
    true /* synchronous */,
    false /* reentrant */,
    [](const std::vector<std::string>& vs, JobResult& job) {
        g_bow.Train(g_bow.Parse(vs[0] + " | " +  vs[1]));
        job.SetPage(Html() << "done");
    },
};

Html Save(const std::string&, const POSTValues&) {
    return Html() << A().Id("dl").Attr("download", "bow_model.bin") << "Download Model" << Close() <<
        Tag("textarea").Id("content").Attr("style", "display: none") << g_bow.Serialize() << Close() <<
        Tag("script") <<
            "window.onload = function() {"
                "var txt = document.getElementById('dl');"
                "txt.href = 'data:text/plain;charset=utf-8,' "
                    "+ encodeURIComponent(document.getElementById('content').value);"
                "};" <<
        Close();
}

Html DisplayWeights(const std::string&, const POSTValues&) {
    Html html;
    for (size_t label = 0; label < g_bow.labels().size(); ++label) {
        html << H2() << g_bow.labels().GetString(label) << Close();

        auto minmax = std::minmax_element(g_bow.weights(label).begin(), g_bow.weights(label).end());
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
        for (size_t w = 0; w < g_bow.GetVocabSize() / 2; ++w) {
            html <<
            Tag("tr") <<
                Tag("td") << g_bow.WordFromId(w) << Close() <<
                Tag("td") <<
                    Div().Attr("style",
                            "width: " + std::to_string(200 * std::abs(g_bow.weight(label, w) / max)) + "px;"
                            "padding-left: 2px;"
                            "background-color: " +
                            std::string(g_bow.weight(label, w) > 0 ? "lightgreen;" : "salmon;")) <<
                        std::to_string(g_bow.weight(label, w) * 100) <<
                    Close() <<
                Close() <<
                Tag("td") << g_bow.WordFromId(w + vocab) << Close() <<
                Tag("td") <<
                    Div().Attr("style",
                            "width: " + std::to_string(200 * std::abs(g_bow.weight(label, w + vocab) / max)) + "px;"
                            "padding-left: 2px;"
                            "background-color: " +
                            std::string(g_bow.weight(label, w + vocab) > 0 ? "lightgreen;" : "salmon;")) <<
                        std::to_string(g_bow.weight(label, w + vocab) * 100) <<
                    Close() <<
                Close() <<
            Close();
        }
        html << Close();
    }
    return html;
}

int main() {
    InitHttpInterface();  // Init the http server
    RegisterJob(classify);
    RegisterJob(train);
    RegisterJob(training_single);
    RegisterJob(load);
    RegisterUrl("/weights", DisplayWeights);
    RegisterUrl("/save", Save);
    ServiceLoopForever();  // infinite loop ending only on SIGINT / SIGTERM / SIGKILL
    StopHttpInterface();  // clear resources
    return 0;
}
