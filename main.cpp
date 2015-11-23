#include <vector>
#include <string>
#include <utility>
#include <map>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <fstream>

#include "bow.h"

#include <httpi/html/html.h>
#include <httpi/html/chart.h>
#include <httpi/html/form-gen.h>
#include <httpi/webjob.h>
#include <httpi/displayer.h>
#include <httpi/monitoring.h>

namespace htmli = httpi::html;

static const unsigned int kNotFound = -1;

BoWClassifier g_bow;

static const htmli::FormDescriptor classify_form_desc = {
    "Classify",
    "Classify the input text to one of the categories",
    { { "input", "text", "Text to classify" } }
};

static const std::string classify_form = classify_form_desc
        .MakeForm("/prediction", "GET").Get();

htmli::Html Classify(const std::string& input) { // the actual function
    using namespace httpi::html;

    std::vector<double> probas(g_bow.OutputSize());
    auto pair = g_bow.ComputeClass(input, probas.data());
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
    return html;
}

static const htmli::FormDescriptor train_form_desc = {
    "Train",
    "Train the model on the specified dataset",
    { { "trainingset", "file", "A training set" },
      { "epochs", "number", "How many iterations?" } }
};

static const std::string train_form = train_form_desc
        .MakeForm("/dataset?action=train", "POST").Get();

class TrainJob : public WebJob {
    size_t nb_epoch_;
    std::string dataset_;

    public:
    TrainJob(size_t nb_epoch, const std::string& dataset)
        : nb_epoch_(nb_epoch), dataset_(dataset) {}

    void Do() {
        Document doc = g_bow.Parse(dataset_);
        htmli::Chart accuracy_chart("accuracy");
        accuracy_chart.Label("iter").Value("accuracy");

        for (size_t epoch = 0; epoch < nb_epoch_; ++epoch) {
            int accuracy = g_bow.Train(doc);

            accuracy_chart.Log("accuracy", accuracy);
            accuracy_chart.Log("iter", epoch);
            SetPage(htmli::Html() << accuracy_chart.Get());
        }
    }
    virtual std::string name() const { return "Train"; }
};

static const htmli::FormDescriptor load_form_desc = {
    "Load",
    "Load a model",
    { { "model", "file", "The model file" } }
};

static const std::string load_form = load_form_desc
        .MakeForm("/model", "POST").Get();

void Load(const std::string& input_str) {
    std::istringstream in(input_str);
    g_bow = BoWClassifier::FromSerialized(in);
}

static const htmli::FormDescriptor training_single_form_desc = {
    "Single example",
    "Add a single training example",
    { { "input", "text", "An input sentence" },
      { "label", "text", "The label" } }
};

static const std::string training_single_form = training_single_form_desc
        .MakeForm("/dataset?action=train_single", "POST").Get();

void TrainSingle(const std::string& example, const std::string& label) {
    g_bow.Train(g_bow.Parse(example + " | " +  label));
}

htmli::Html Save() {
    using namespace httpi::html;
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

htmli::Html DisplayWeights() {
    using namespace httpi::html;

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

std::string MakePage(const std::string& content) {
    using namespace httpi::html;
    return (httpi::html::Html() <<
        "<!DOCTYPE html>"
        "<html>"
           "<head>"
                R"(<meta charset="utf-8">)"
                R"(<meta http-equiv="X-UA-Compatible" content="IE=edge">)"
                R"(<meta name="viewport" content="width=device-width, initial-scale=1">)"
                R"(<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap.min.css">)"
                R"(<link rel="stylesheet" href="//cdn.jsdelivr.net/chartist.js/latest/chartist.min.css">)"
                R"(<script src="//cdn.jsdelivr.net/chartist.js/latest/chartist.min.js"></script>)"
            "</head>"
            "<body lang=\"en\">"
                "<div class=\"container\">"
                    "<div class=\"col-md-9\">" <<
                        content <<
                    "</div>"
                    "<div class=\"col-md-3\">" <<
                        H2() << "Go to" << Close() <<
                        Ul() <<
                            Li() <<
                                A().Attr("href", "/jobs") <<
                                    "Jobs" <<
                                Close() <<
                            Close() <<
                            Li() <<
                                A().Attr("href", "/prediction") <<
                                    "Prediction" <<
                                Close() <<
                            Close() <<
                            Li() <<
                                A().Attr("href", "/dataset") <<
                                    "Dataset" <<
                                Close() <<
                            Close() <<
                            Li() <<
                                A().Attr("href", "/model") <<
                                    "Model" <<
                                Close() <<
                            Close() <<
                        Close() <<
                    "</div>"
                "</div>"
            "</body>"
        "</html>").Get();
}

int main() {
    InitHttpInterface();  // Init the http server

    WebJobsPool jp;
    auto t1 = jp.StartJob(std::unique_ptr<MonitoringJob>(new MonitoringJob()));
    auto monitoring_job = jp.GetId(t1);

    RegisterUrl("/", [&monitoring_job](const std::string&, const POSTValues&) {
            return MakePage(*monitoring_job->job_data().page());
        });

    RegisterUrl("/prediction",
            [](const std::string&, const POSTValues& args) {
                using namespace httpi::html;
                Html html;

                auto vargs = classify_form_desc.ValidateParams(args);
                if (std::get<0>(vargs)) {
                    html << P().AddClass("alert") << "No input" << Close();
                } else {
                    html << Classify(std::get<2>(vargs)[0]);
                }
                html << classify_form;
                return MakePage(html.Get());
            });

    RegisterUrl("/dataset",
            [&jp](const std::string& method, const POSTValues& args) {
                htmli::Html html;

                auto action_iter = args.find("action");
                std::string action = action_iter == args.end()
                        ? "none" : action_iter->second;
                if (action == "train") {
                    auto vargs = train_form_desc.ValidateParams(args);
                    if (std::get<0>(vargs)) {
                        html << std::get<1>(vargs);
                    } else {
                        jp.StartJob(std::unique_ptr<TrainJob>(
                                new TrainJob(
                                    std::atoi(std::get<2>(vargs)[1].c_str()),
                                    std::get<2>(vargs)[0])));
                        html << "learning started";
                    }
                } else if (action == "train_single") {
                    auto vargs = training_single_form_desc.ValidateParams(args);
                    if (std::get<0>(vargs)) {
                        html << std::get<1>(vargs);
                    } else {
                        auto& form_args = std::get<2>(vargs);
                        TrainSingle(form_args[0], form_args[1]);
                        html << "learning started";
                    }
                }
                html << train_form << training_single_form;
                return MakePage(html.Get());
            });

    RegisterUrl("/model",
            [](const std::string& method, const POSTValues& args) {
                htmli::Html html;
                html << Save();
                if (method == "POST") {
                    auto vargs = load_form_desc.ValidateParams(args);
                    if (std::get<0>(vargs)) {
                        html << std::get<1>(vargs);
                    } else {
                        Load(std::get<2>(vargs)[0]);
                        html << "learning started";
                    }
                }
                html << load_form;
                html << DisplayWeights();
                return MakePage(html.Get());
            });

    RegisterUrl("/jobs", [&jp](const std::string&, const POSTValues& args) {
            using namespace httpi::html;
            auto id = args.find("id");
            if (id == args.end()) {
                Html html;
                html <<
                    Table().AddClass("table") <<
                        Tr() <<
                            Th() << "Job" << Close() <<
                            Th() << "Started" << Close() <<
                            Th() << "Finished" << Close() <<
                            Th() << "Details" << Close() <<
                        Close();

                jp.foreach_job([&html](WebJobsPool::job_type& x) {
                    html <<
                        Tr() <<
                            Td() <<
                                std::to_string(x.first) << ": " <<
                                x.second->job_data().name() <<
                            Close() <<
                            Td() << "xxxx" << Close() <<
                            Td() <<
                                    (x.second->IsFinished() ? "true" : "false") <<
                            Close() <<
                            Td() <<
                                A().Attr("href", "/jobs?id="
                                    + std::to_string(x.first)) << "See" <<
                                Close() <<
                            Close() <<
                        Close();
                });

                html << Close();
                return MakePage(html.Get());
            } else {
                Html html;
                auto job = jp.GetId(std::atoi(id->second.c_str()));

                if (job == nullptr) {
                    return MakePage((Html() << "not found").Get());
                }

                return MakePage(*job->job_data().page());
            }
        });

    ServiceLoopForever();  // infinite loop ending only on SIGINT / SIGTERM / SIGKILL
    StopHttpInterface();  // clear resources
    return 0;
}
