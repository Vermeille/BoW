#include <iostream>
#include <fstream>

#include <nlp/dict.h>
#include <nlp/tokenizer.h>
#include <nlp/bow.h>

Document Parse(const std::string& str, NGramMaker& ngram, LabelSet& ls) {
    std::ifstream dataset(str);
    std::string line;
    Document doc;
    while (std::getline(dataset, line)) {
        size_t pipe = line.find('|');
        std::string data(line, 0, pipe - 1);
        std::string label(line, pipe + 2, line.size());

        std::vector<WordFeatures> toks = Tokenizer::FR(data);
        ngram.Learn(toks);
        doc.examples.push_back(TrainingExample{toks, ls.GetLabel(label)});
    }
    return doc;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset>\n";
        return EXIT_FAILURE;
    }

    NGramMaker ngram;
    BagOfWords bow(200, 2);
    LabelSet ls;
    Document doc = Parse(argv[1], ngram, ls);

    bow.ResizeInput(ngram.dict().size());
    bow.ResizeOutput(ls.size());

    std::cout << "Training...\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << bow.Train(doc) << "% accuracy" << std::endl;
    }

    std::cout << "> ";
    std::string line;
    while (std::getline(std::cin, line)) {
        std::vector<WordFeatures> toks = Tokenizer::FR(line);
        ngram.Annotate(toks);
        auto prediction = bow.ComputeClass(toks);

        for (size_t l = 0; l < ls.size(); ++l) {
            std::cout << ls.GetString(l) << ": " << prediction(l, 0) << "\n";
        }
        std::cout << "> ";
    }

    return 0;
}
