#include "pages.h"

httpi::html::Html DisplayWeights(BoWClassifier& bow) {
    using namespace httpi::html;

    Html html;
    for (size_t label = 0; label < bow.labels().size(); ++label) {
        html << H2() << bow.labels().GetString(label) << Close();

        auto label_row = bow.weights().row(label);
        double max = std::max(
                std::abs(label_row.minCoeff()),
                std::abs(label_row.maxCoeff()));

        html <<
        Tag("table").AddClass("table") <<
            Tag("tr") <<
                Tag("th") << "Word" << Close() <<
                Tag("th") << "Score" << Close() <<
                Tag("th") << "Word" << Close() <<
                Tag("th") << "Score" << Close() <<
            Close();

        int vocab = bow.GetVocabSize() / 2;
        for (size_t w = 0; w < bow.GetVocabSize() / 2; ++w) {
            html <<
            Tag("tr") <<
                Tag("td") << bow.WordFromId(w) << Close() <<
                Tag("td") <<
                    Div().Attr("style",
                            "width: " + std::to_string(200 * std::abs(bow.weights(label, w) / max)) + "px;"
                            "padding-left: 2px;"
                            "background-color: " +
                            std::string(bow.weights(label, w) > 0 ? "lightgreen;" : "salmon;")) <<
                        std::to_string(bow.weights(label, w) * 100) <<
                    Close() <<
                Close() <<
                Tag("td") << bow.WordFromId(w + vocab) << Close() <<
                Tag("td") <<
                    Div().Attr("style",
                            "width: " + std::to_string(200 * std::abs(bow.weights(label, w + vocab) / max)) + "px;"
                            "padding-left: 2px;"
                            "background-color: " +
                            std::string(bow.weights(label, w + vocab) > 0 ? "lightgreen;" : "salmon;")) <<
                        std::to_string(bow.weights(label, w + vocab) * 100) <<
                    Close() <<
                Close() <<
            Close();
        }
        html << Close();
    }
    return html;
}
