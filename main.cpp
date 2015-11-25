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

#include <httpi/html/html.h>
#include <httpi/html/form-gen.h>
#include <httpi/html/chart.h>
#include <httpi/webjob.h>
#include <httpi/displayer.h>
#include <httpi/monitoring.h>

namespace htmli = httpi::html;

static const htmli::FormDescriptor train_form_desc = {
    "Train",
    "Train the model on the specified dataset",
    { { "trainingset", "file", "A training set" },
      { "epochs", "number", "How many iterations?" } }
};

static const std::string train_form = train_form_desc
        .MakeForm("/dataset?action=train", "POST").Get();

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

static const htmli::FormDescriptor load_form_desc = {
    "Load",
    "Load a model",
    { { "model", "file", "The model file" } }
};

static const std::string load_form = load_form_desc
        .MakeForm("/model", "POST").Get();

BoWClassifier Load(const std::string& input_str) {
    std::istringstream in(input_str);
    return BoWClassifier::FromSerialized(in);
}

static const htmli::FormDescriptor training_single_form_desc = {
    "Single example",
    "Add a single training example",
    { { "input", "text", "An input sentence" },
      { "label", "text", "The label" } }
};

static const std::string training_single_form = training_single_form_desc
        .MakeForm("/dataset?action=train_single", "POST").Get();

void TrainSingle(BoWClassifier& bow, const std::string& example,
        const std::string& label) {
    bow.Train(bow.Parse(example + " | " +  label));
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

    RegisterUrl("/prediction",
            std::bind(Classify,
                std::ref(bow),
                std::placeholders::_1,
                std::placeholders::_2)
            );

    RegisterUrl("/dataset",
            [&jp, &bow](const std::string& method, const POSTValues& args) {
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
                                    bow,
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
                        TrainSingle(bow, form_args[0], form_args[1]);
                        html << "learning started";
                    }
                }
                html << train_form << training_single_form;
                return PageGlobal(html.Get());
            });

    RegisterUrl("/model",
            [&bow](const std::string& method, const POSTValues& args) {
                htmli::Html html;
                html << Save(bow);
                if (method == "POST") {
                    auto vargs = load_form_desc.ValidateParams(args);
                    if (std::get<0>(vargs)) {
                        html << std::get<1>(vargs);
                    } else {
                        bow = Load(std::get<2>(vargs)[0]);
                        html << "learning started";
                    }
                }
                html << load_form;
                html << DisplayWeights(bow);
                return PageGlobal(html.Get());
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
