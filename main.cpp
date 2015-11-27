#include <vector>
#include <string>
#include <utility>
#include <map>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <functional>

#include "bow.h"
#include "pages/pages.h"

#include <httpi/rest-helpers.h>
#include <httpi/html/html.h>
#include <httpi/html/json.h>
#include <httpi/html/form-gen.h>
#include <httpi/html/chart.h>
#include <httpi/webjob.h>
#include <httpi/displayer.h>
#include <httpi/monitoring.h>

namespace htmli = httpi::html;

class TrainJob : public WebJob {
    BoWClassifier& bow_;
    size_t nb_epoch_;
    std::string dataset_;

    public:
    TrainJob(BoWClassifier& bow, size_t nb_epoch, const std::string& dataset)
        : bow_(bow), nb_epoch_(nb_epoch), dataset_(dataset) {}

    void Do() {
        Document doc = bow_.Parse(dataset_);
        htmli::Chart accuracy_chart("accuracy");
        accuracy_chart.Label("iter").Value("accuracy");

        for (size_t epoch = 0; epoch < nb_epoch_; ++epoch) {
            int accuracy = bow_.Train(doc);

            accuracy_chart.Log("accuracy", accuracy);
            accuracy_chart.Log("iter", epoch);
            SetPage(htmli::Html() << accuracy_chart.Get());
        }
    }
    virtual std::string name() const { return "Train"; }
};

void TrainSingle(BoWClassifier& bow, const std::string& example,
        const std::string& label) {
    bow.Train(bow.Parse(example + " | " +  label));
}

BoWClassifier Load(const std::string& input_str) {
    std::istringstream in(input_str);
    return BoWClassifier::FromSerialized(in);
}

htmli::Html Save(const BoWClassifier& bow) {
    using namespace httpi::html;
    return Html() << A().Id("dl").Attr("download", "bow_model.bin") << "Download Model" << Close() <<
        Tag("textarea").Id("content").Attr("style", "display: none") << bow.Serialize() << Close() <<
        Tag("script") <<
            "window.onload = function() {"
                "var txt = document.getElementById('dl');"
                "txt.href = 'data:text/plain;charset=utf-8,' "
                    "+ encodeURIComponent(document.getElementById('content').value);"
                "};" <<
        Close();
}

int main() {
    InitHttpInterface();  // Init the http server

    WebJobsPool jp;
    auto t1 = jp.StartJob(std::unique_ptr<MonitoringJob>(new MonitoringJob()));
    auto monitoring_job = jp.GetId(t1);

    BoWClassifier bow;

    RegisterUrl("/", [&monitoring_job](const std::string&, const POSTValues&) {
            return PageGlobal(*monitoring_job->job_data().page());
        });

    RegisterUrl("/prediction", httpi::RestPageMaker(PageGlobal)
        .AddResource("GET", httpi::RestResource(
            htmli::FormDescriptor<std::string> {
                "GET", "/prediction",
                "Classify",
                "Classify the input text to one of the categories",
                { { "input", "text", "Text to classify" } }
            },
            [&bow](const std::string& input) {
                return bow.ComputeClass(input);
            },
            [&bow](const BowResult& res) {
                return ClassifyResult(bow, res);
            },
            [&bow](const BowResult& res) {
                return JsonBuilder()
                    .Append("confidence", res.confidence[res.label])
                    .Append("label", bow.labels().GetString(res.label))
                    .Build();
            })));

    RegisterUrl("/model", httpi::RestPageMaker(PageGlobal)
        .AddResource("POST", httpi::RestResource(
            htmli::FormDescriptor<std::string> {
                "POST", "/model",
                "Load",
                "Load a model",
                { { "model", "file", "The model file" } }
            },
            [&bow](const std::string& model) {
                htmli::Html html;
                html << Save(bow);
                bow = Load(model);
                return 0;
            },
            [](int) {
                return htmli::Html() << "Model loaded";
            },
            [](int) {
                return JsonBuilder().Append("result", 0).Build();
            }))
        .AddResource("GET", httpi::RestResource(
            htmli::FormDescriptor<>{},
            [](){ return 0; },
            [&bow](int) {
                return Save(bow) << DisplayWeights(bow);
            },
            [](int) {
                // FIXME: not implemented. The model is serialized with
                // newlines and multilines strings are forbidden in JSON.
                return JsonBuilder().Append("result", 1).Build();
            })));

    RegisterUrl("/dataset", httpi::RestPageMaker(PageGlobal)
        .AddResource("PUT", httpi::RestResource(
            htmli::FormDescriptor<std::string, std::string> {
                "PUT", "/dataset",
                "Single example",
                "Add a single training example",
                { { "input", "text", "An input sentence" },
                  { "label", "text", "The label" } }
            },
            [&jp, &bow](const std::string& input, const std::string& label) {
                TrainSingle(bow, input, label);
                return 0;
            },
            [](int) { return htmli::Html() << "Learning started"; },
            [](int) { return JsonBuilder().Append("result", 0).Build(); }))
        .AddResource("POST", httpi::RestResource(
            htmli::FormDescriptor<std::string, int> {
                "POST", "/dataset",
                "Train",
                "Train the model on the specified dataset",
                { { "trainingset", "file", "A training set" },
                  { "epochs", "number", "How many iterations?" } }
            },
            [&jp, &bow](const std::string& trainingset, int epoch) {
                return jp.StartJob(
                        std::make_unique<TrainJob>(bow, epoch, trainingset));
            },
            [](int id) {
                using namespace htmli;
                return Html() <<
                    A().Attr("href", "/jobs?id=" + std::to_string(id)) <<
                        "Learning started" <<
                    Close();
            },
            [](int id) {
                return JsonBuilder().Append("job_id", id).Build();
            }))
        .AddResource("GET", httpi::RestResource(
                htmli::FormDescriptor<> {},
                [](){ return 0; },
                [](int){ return htmli::Html();},
                [](int){ return ""; }
            )));

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
                return PageGlobal(html.Get());
            } else {
                Html html;
                auto job = jp.GetId(std::atoi(id->second.c_str()));

                if (job == nullptr) {
                    return PageGlobal((Html() << "not found").Get());
                }

                return PageGlobal(*job->job_data().page());
            }
        });

    ServiceLoopForever();  // infinite loop ending only on SIGINT / SIGTERM / SIGKILL
    StopHttpInterface();  // clear resources
    return 0;
}
