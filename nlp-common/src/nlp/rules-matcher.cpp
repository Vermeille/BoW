#include "rules-matcher.h"

#include <sstream>

std::vector<std::string> RulesMatcher::Match(const TrainingExample& str) const {
    std::vector<std::string> matches;

    for (auto& r : rules_) {
        if (r.Matches(str)) {
            matches.push_back(r.key());
        }
    }
    return matches;
}

RulesMatcher RulesMatcher::FromSerialized(const std::string& str) {
    RulesMatcher rm;
    std::istringstream iss(str);
    std::string line;
    while (std::getline(iss, line)) {
        rm.AddRule(line);
    }
    return rm;
}

std::string RulesMatcher::Serialize() const {
    std::ostringstream oss;
    for (auto& r : rules_) {
        oss << r.key() << " : " << r.AsString() << std::endl;
    }
    return oss.str();
}
