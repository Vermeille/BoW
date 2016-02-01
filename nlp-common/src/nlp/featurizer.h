#pragma once

#include <string>
#include <vector>

#include "document.h"

struct FeaturesExtractor {
    static std::vector<WordFeatures> Do(const std::vector<std::string>& sentence);
};


