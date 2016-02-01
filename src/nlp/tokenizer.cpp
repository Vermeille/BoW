#include <locale>

#include "tokenizer.h"

std::vector<WordFeatures> Tokenizer::FR(const std::string& str) {
    std::vector<WordFeatures> sentence;
    size_t idx = 0;

    std::locale fr("fr_FR.UTF-8");

    std::string tok;
    while (idx < str.size()) {
        if (isspace(str[idx], fr)) {
            // skip & end word
            if (tok != "") {
                sentence.emplace_back(tok);
                tok.clear();
            }
        } else if (isalnum(str[idx], fr) || str[idx] == '-') {
            tok += str[idx];
        } else if (str[idx] == '\'') {
            tok += '\'';
            sentence.emplace_back(tok);
            tok.clear();
        } else if (ispunct(str[idx], fr)) {
            sentence.emplace_back(tok);
            tok.clear();
            sentence.emplace_back(std::string() + str[idx]);
        }

        ++idx;
    }
    if (tok != "") {
        sentence.emplace_back(tok);
    }
    return sentence;
}
