#include "pages.h"

httpi::html::Html DisplayWeights(BoWClassifier& bow) {
    using namespace httpi::html;

    Html html;
    for (size_t label = 0; label < bow.labels().size(); ++label) {
        html << H2() << bow.labels().GetString(label) << Close();

        auto minmax = std::minmax_element(bow.weights(label).begin(), bow.weights(label).end());
        double max = std::max(std::abs(*minmax.first), std::abs(*minmax.second));

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
                            "width: " + std::to_string(200 * std::abs(bow.weight(label, w) / max)) + "px;"
                            "padding-left: 2px;"
                            "background-color: " +
                            std::string(bow.weight(label, w) > 0 ? "lightgreen;" : "salmon;")) <<
                        std::to_string(bow.weight(label, w) * 100) <<
                    Close() <<
                Close() <<
                Tag("td") << bow.WordFromId(w + vocab) << Close() <<
                Tag("td") <<
                    Div().Attr("style",
                            "width: " + std::to_string(200 * std::abs(bow.weight(label, w + vocab) / max)) + "px;"
                            "padding-left: 2px;"
                            "background-color: " +
                            std::string(bow.weight(label, w + vocab) > 0 ? "lightgreen;" : "salmon;")) <<
                        std::to_string(bow.weight(label, w + vocab) * 100) <<
                    Close() <<
                Close() <<
            Close();
        }
        html << Close();
    }
    return html;
}
