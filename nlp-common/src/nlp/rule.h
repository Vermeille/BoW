#pragma once

#include <vector>
#include <string>

#include "document.h"

class Rule {
    std::string key_;
    std::vector<std::string> pattern_;

    bool Matches(const TrainingExample& ex, size_t pidx, size_t exidx) const;

  public:
    Rule(const std::string& str);
    bool Matches(const TrainingExample& ex) const;
    const std::string& key() const { return key_; }

    std::string AsString() const;
};
