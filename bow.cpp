#include <fstream>
#include <glog/logging.h>

#include <nlp-common/tokenizer.h>

#include "bow.h"

size_t BoWClassifier::Train(const Document& doc) {

    bow_.ResizeInput(ngram_.dict().size());
    bow_.ResizeOutput(ls_.size());

    return bow_.Train(doc);
}

Document BoWClassifier::Parse(const std::string& str) {
    std::istringstream dataset(str);
    std::string line;
    Document doc;
    while (std::getline(dataset, line)) {
        size_t pipe = line.find('|');
        std::string data(line, 0, pipe - 1);
        std::string label(line, pipe + 2, line.size());

        std::vector<WordFeatures> toks = Tokenizer::FR(data);
        ngram_.Learn(toks);
        doc.examples.push_back(TrainingExample{toks, ls_.GetLabel(label)});
    }
    return doc;
}

std::pair<Label, std::vector<WordFeatures>>
BoWClassifier::ComputeClass(const std::string& data, double *probas) {
    auto toks = Tokenizer::FR(data);
    ngram_.Annotate(toks);

    return std::make_pair(bow_.ComputeClass(toks, probas), toks);
}

