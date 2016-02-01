#include "featurizer.h"

std::vector<WordFeatures> FeaturesExtractor::Do(const std::vector<std::string>& sentence) {
    std::vector<WordFeatures> fs;
    for (auto& s : sentence) {
        fs.emplace_back(s);
    }
    return fs;
}
