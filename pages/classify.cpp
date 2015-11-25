#include "pages.h"

#include <httpi/html/form-gen.h>

static const unsigned int kNotFound = -1;

static const httpi::html::FormDescriptor classify_form_desc = {
    "Classify",
    "Classify the input text to one of the categories",
    { { "input", "text", "Text to classify" } }
};

static const std::string classify_form = classify_form_desc
        .MakeForm("/prediction", "GET").Get();

static httpi::html::Html ClassifyResult(BoWClassifier& bow, const std::string& input) {
    using namespace httpi::html;

    std::vector<double> probas(bow.OutputSize());
    auto pair = bow.ComputeClass(input, probas.data());
    Label k = pair.first;
    auto& ws = pair.second;

    // Header
    Html html;
    for (auto w : ws) {
        if (w.idx != kNotFound) {
            html <<
                Tag("span").Attr("style",
                    "font-size: " + std::to_string((1 + std::log(1 + std::abs(bow.weight(k, w.idx)))) * 30) + "px;"
                    "color: " + std::string(bow.weight(k, w.idx) > 0 ? "green" : "red") + ";")
                    << bow.WordFromId(w.idx) << " " << Close();
        } else {
            html << Tag("span") << "_UNK_ " << Close();
        }
    }
    html <<
        P() << "best prediction: " << bow.labels().GetString(k) << " " << std::to_string(probas[k] * 100)
            << Close() <<

    // table of global confidence
        Tag("table").AddClass("table") <<
            Tag("tr") <<
                Tag("th") << "Label" << Close() <<
                Tag("th") << "Confidence" << Close() <<
            Close();

    for (size_t i = 0; i < bow.labels().size(); ++i) {
        html <<
        Tag("tr") <<
            Tag("td") << bow.labels().GetString(i) << Close() <<
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

std::string Classify(BoWClassifier& bow, const std::string&,
        const POSTValues& args) {
    using namespace httpi::html;
    Html html;

    auto vargs = classify_form_desc.ValidateParams(args);
    if (std::get<0>(vargs)) {
        html << P().AddClass("alert") << "No input" << Close();
    } else {
        html << ClassifyResult(bow, std::get<2>(vargs)[0]);
    }
    html << classify_form;
    return PageGlobal(html.Get());
}
