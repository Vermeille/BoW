#include <fstream>
#include <glog/logging.h>

#include <nlp/tokenizer.h>

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

BowResult BoWClassifier::ComputeClass(const std::string& data) {
    auto toks = Tokenizer::FR(data);
    ngram_.Annotate(toks);

    Eigen::MatrixXd probas = bow_.ComputeClass(toks);
    Eigen::MatrixXd::Index label_res, dummy_zero;
    probas.maxCoeff(&label_res, &dummy_zero);
    return {probas, label_res, toks};
}

BoWClassifier BoWClassifier::FromSerialized(std::istream& in) {
    BoWClassifier bow;
    bow.ngram_ = NGramMaker::FromSerialized(in);
    bow.bow_ = BagOfWords::FromSerialized(in);
    bow.ls_ = LabelSet::FromSerialized(in);
    return bow;
}
