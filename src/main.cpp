#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "bow.h"
#include "pages/pages.h"

#include <httpi/displayer.h>
#include <httpi/html/chart.h>
#include <httpi/html/form-gen.h>
#include <httpi/html/html.h>
#include <httpi/html/json.h>
#include <httpi/monitoring.h>
#include <httpi/rest-helpers.h>
#include <httpi/webjob.h>

namespace htmli = httpi::html;

class TrainJob : public WebJob {
    BoWClassifier& bow_;
    size_t nb_epoch_;
    const Document& trainingset_;
    bool stopped_;

   public:
    TrainJob(BoWClassifier& bow, const Document& ts, size_t nb_epoch)
        : bow_(bow), nb_epoch_(nb_epoch), trainingset_(ts), stopped_(false) {}

    void Do() {
        htmli::Chart accuracy_chart("accuracy");
        accuracy_chart.Label("iter").Value("accuracy");

        for (size_t epoch = 0; epoch < nb_epoch_ && !stopped_; ++epoch) {
            int accuracy = bow_.Train(trainingset_);

            accuracy_chart.Log("accuracy", accuracy);
            accuracy_chart.Log("iter", epoch);
            SetPage(htmli::Html() << accuracy_chart.Get());
        }
    }
    virtual std::string name() const { return "Train"; }

    virtual void Stop() { stopped_ = true; }
};

void AddExample(BoWClassifier& bow,
                Document& ts,
                const std::string& example,
                const std::string& label,
                size_t nb_epoch) {
    Document minibatch = bow.Parse(example + " | " + label);
    ts.examples.push_back(minibatch.examples[0]);
    std::copy_n(ts.examples.rbegin(),
                std::min(ts.examples.size(), 9ul),
                std::back_inserter(minibatch.examples));
    for (size_t epoch = 0; epoch < nb_epoch; ++epoch) {
        bow.Train(minibatch);
    }
}

BoWClassifier Load(const std::string& input_str) {
    std::istringstream in(input_str);
    return BoWClassifier::FromSerialized(in);
}

htmli::Html Save(const BoWClassifier& bow) {
    using namespace httpi::html;
    // clang-format off
    return Html() << A().Id("dl").Attr("download", "bow_model.bin") << "Download Model" << Close() <<
        Tag("textarea").Id("content").Attr("style", "display: none") << bow.Serialize() << Close() <<
        Tag("script") <<
            "window.onload = function() {"
                "var txt = document.getElementById('dl');"
                "txt.href = 'data:text/plain;charset=utf-8,' "
                    "+ encodeURIComponent(document.getElementById('content').value);"
                "};" <<
        Close();
    // clang-format on
}

std::string SerializeDataset(BoWClassifier& bow, const Document& doc) {
    std::ostringstream out;
    for (auto& ex : doc.examples) {
        for (auto& w : ex.inputs) {
            out << w.str << " ";
        }
        out << "| " << bow.labels().GetString(ex.output) << std::endl;
    }
    return out.str();
}

htmli::Html SaveDataset(BoWClassifier& bow, const Document& doc) {
    using namespace httpi::html;
    return Html()
           << A().Id("dl").Attr("download", "bow_dataset.bin")
           << "Download dataset" << Close()
           << Tag("textarea").Id("content").Attr("style", "display: none")
           << SerializeDataset(bow, doc) << Close() << Tag("script")
           << "window.onload = function() {"
              "var txt = document.getElementById('dl');"
              "txt.href = 'data:text/plain;charset=utf-8,' "
              "+ encodeURIComponent(document.getElementById('content').value);"
              "};"
           << Close();
}

int main() {
    HTTPServer server(8080);
    WebJobsPool jp;
    auto t1 =
        jp.StartJob(std::unique_ptr<MonitoringJob>(new MonitoringJob(30, 30)));
    auto monitoring_job = jp.GetId(t1);

    BoWClassifier bow;
    Document trainingset;

    server.RegisterUrl(
        "/", [&monitoring_job](const std::string&, const POSTValues&) {
            return PageGlobal(*monitoring_job->job_data().page());
        });

    server.RegisterUrl(
        "/prediction",
        httpi::RestPageMaker(PageGlobal)
            .AddResource(
                "GET",
                httpi::RestResource(
                    htmli::FormDescriptor<std::string>{
                        "GET",
                        "/prediction",
                        "Classify",
                        "Classify the input text to one of the categories",
                        {{"input", "text", "Text to classify"}}},
                    [&bow](const std::string& input) {
                        return bow.ComputeClass(input);
                    },
                    [&bow](const BowResult& res) {
                        return ClassifyResult(bow, res);
                    },
                    [&bow](const BowResult& res) {
                        return JsonBuilder()
                            .Append("confidence", res.confidence(res.label, 0))
                            .Append("label", bow.labels().GetString(res.label))
                            .Build();
                    })));

    server.RegisterUrl(
        "/model",
        httpi::RestPageMaker(PageGlobal)
            .AddResource(
                "POST",
                httpi::RestResource(
                    htmli::FormDescriptor<std::string>{
                        "POST",
                        "/model",
                        "Load",
                        "Load a model",
                        {{"model", "file", "The model file"}}},
                    [&bow](const std::string& model) {
                        htmli::Html html;
                        html << Save(bow);
                        bow = Load(model);
                        return 0;
                    },
                    [](int) { return htmli::Html() << "Model loaded"; },
                    [](int) {
                        return JsonBuilder().Append("result", 0).Build();
                    }))
            .AddResource(
                "GET",
                httpi::RestResource(
                    htmli::FormDescriptor<>{},
                    []() { return 0; },
                    [&bow](int) { return Save(bow) << DisplayWeights(bow); },
                    [](int) {
                        // FIXME: not implemented. The model is serialized with
                        // newlines and multilines strings are forbidden in
                        // JSON.
                        return JsonBuilder().Append("result", 1).Build();
                    })));

    server.RegisterUrl(
        "/dataset",
        httpi::RestPageMaker(PageGlobal)
            .AddResource(
                "PUT",
                httpi::RestResource(
                    htmli::FormDescriptor<std::string, std::string, int>{
                        "PUT",
                        "/dataset",
                        "Single example",
                        "Add a single training example",
                        {{"input", "text", "An input sentence"},
                         {"label", "text", "The label"},
                         {"epoch", "number", "How many iterations?"}}},
                    [&jp, &bow, &trainingset](const std::string& input,
                                              const std::string& label,
                                              int epoch) {
                        AddExample(bow, trainingset, input, label, epoch);
                        return 0;
                    },
                    [](int) { return htmli::Html() << "Learning started"; },
                    [](int) {
                        return JsonBuilder().Append("result", 0).Build();
                    }))
            .AddResource(
                "POST",
                httpi::RestResource(
                    htmli::FormDescriptor<std::string, int>{
                        "POST",
                        "/dataset",
                        "Upload dataset",
                        "Uploads a new dataset",
                        {{"trainingset", "file", "A training set"},
                         {"epoch", "number", "Number of training epochs"}}},
                    [&jp, &bow, &trainingset](
                        const std::string& str_trainingset, int epoch) {
                        trainingset = bow.Parse(str_trainingset);
                        return jp.StartJob(std::make_unique<TrainJob>(
                            bow, trainingset, epoch));
                    },
                    [](int id) {
                        using namespace htmli;
                        return Html()
                               << A().Attr("href",
                                           "/jobs?id=" + std::to_string(id))
                               << "Learning started" << Close();
                    },
                    [](int id) {
                        return JsonBuilder().Append("job_id", id).Build();
                    }))
            .AddResource("GET",
                         httpi::RestResource(htmli::FormDescriptor<>{},
                                             []() { return 0; },
                                             [&bow, &trainingset](int) {
                                                 return SaveDataset(
                                                     bow, trainingset);
                                             },
                                             [](int) { return ""; })));

    server.RegisterUrl(
        "/jobs", [&jp](const std::string&, const POSTValues& args) {
            using namespace httpi::html;
            auto id = args.find("id");
            if (id == args.end()) {
                // clang-format off
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
                // clang-format on
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

    server.ServiceLoopForever();
    return 0;
}
