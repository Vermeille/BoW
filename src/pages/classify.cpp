#include "pages.h"

#include <httpi/html/form-gen.h>

static const unsigned int kNotFound = -1;

httpi::html::Html ClassifyResult(BoWClassifier& bow, const BowResult& bowr) {
    using namespace httpi::html;
    Label k = bowr.label;
    auto& probas = bowr.confidence;

    // Header
    Html html;
    for (auto w : bowr.words) {
        if (w.idx != kNotFound) {
            html <<
                Tag("span").Attr("style",
                    "font-size: " + std::to_string((1 + std::log(1 + std::abs(bow.weights(k, w.idx)))) * 30) + "px;"
                    "color: " + std::string(bow.weights(k, w.idx) > 0 ? "green" : "red") + ";")
                    << bow.WordFromId(w.idx) << " " << Close();
        } else {
            html << Tag("span") << "_UNK_ " << Close();
        }
    }
    html <<
        P() << "best prediction: " << bow.labels().GetString(k) <<
        " " << std::to_string(probas(k, 0) * 100)
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
                        "width: " + std::to_string(200 * probas(i, 0)) + "px;"
                        "padding-left: 2px;"
                        "background-color: lightgreen;") <<
                    std::to_string(probas(i, 0) * 100) <<
                Close() <<
            Close() <<
        Close();
    }
    html << Close();
    return html;
}

