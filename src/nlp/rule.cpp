#include "rule.h"

#include <sstream>
#include <glog/logging.h>

Rule::Rule(const std::string& str) {
    std::istringstream iss(str);

    iss >> key_;

    std::string w;
    iss >> w;
    LOG_IF(FATAL, w != ":") << "Bad syntax";

    iss >> w;
    while (iss) {
        pattern_.push_back(w);
        iss >> w;
    }
}

bool Rule::Matches(const TrainingExample& ex) const {
    if (pattern_[0] == "^") {
        return Matches(ex, 1, 0);
    } else {
        for (size_t i = 0; i < ex.inputs.size(); ++i) {
            if (Matches(ex, 0, i)) {
                return true;
            }
        }
        return false;
    }
}

bool Rule::Matches(const TrainingExample& ex, size_t pidx, size_t exidx) const {
    if (pattern_.size() <= pidx) {
        // pattern consumed
        return true;
    }

    if (pattern_[pidx] == "$") {
        // check for end
        return exidx == ex.inputs.size();
    }

    if (ex.inputs.size() <= exidx) {
        // input too short
        return false;
    }

    if (pattern_[pidx] == ex.inputs[exidx].str) {
        // next word
        return Matches(ex, pidx + 1, exidx + 1);
    }

    if (pattern_[pidx] == "_") {
        // next word
        return Matches(ex, pidx + 1, exidx + 1);
    }

    if (pattern_[pidx] == "*") {
        while (exidx < ex.inputs.size()) {
            if (Matches(ex, pidx + 1, exidx)) {
                return true;
            }
            ++exidx;
        }
        return false;
    }

    return false;
}

std::string Rule::AsString() const {
    if (pattern_.size() == 0) {
        return "";
    }

    std::string str = pattern_[0];
    for (size_t i = 1; i < pattern_.size(); ++i) {
        str += " " + pattern_[i];
    }
    return str;
}
