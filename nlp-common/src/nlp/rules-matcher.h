#pragma once

#include <vector>
#include <string>

#include "rule.h"

class RulesMatcher {
    std::vector<Rule> rules_;

  public:
    void AddRule(const std::string& str) { rules_.emplace_back(str); }

    std::vector<std::string> Match(const TrainingExample& str) const;

    size_t size() const { return rules_.size(); }

    std::vector<Rule>::iterator begin() { return rules_.begin(); }
    std::vector<Rule>::iterator end() { return rules_.end(); }

    static RulesMatcher FromSerialized(const std::string& str);

    std::string Serialize() const;
};

