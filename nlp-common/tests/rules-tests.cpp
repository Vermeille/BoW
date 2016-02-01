#include <sstream>
#include <string>
#include <iostream>

#include <nlp/rules-matcher.h>

TrainingExample Parse(const std::string& str) {
    std::istringstream input(str);
    std::string w;
    TrainingExample ex;
    input >> w;
    while (input) {
        //ex.inputs.push_back({0, w});
        input >> w;
    }
    return ex;
}

int main() {
    RulesMatcher rm;
    rm.AddRule("HELLO : ^ bonjour");
    rm.AddRule("CAVA : ça va");
    std::cout << (rm.Match(Parse("bonjour michel")).size() == 1)    << std::endl;
    std::cout << (rm.Match(Parse("bonjour")).size() == 1)           << std::endl;
    std::cout << (rm.Match(Parse("eh bonjour")).size() == 0)        << std::endl;
    std::cout << (rm.Match(Parse("comment ça va ?")).size() == 1)   << std::endl;
    std::cout << (rm.Match(Parse("comment ça")).size() == 0)        << std::endl;
    std::cout << (rm.Match(Parse("comment ça caca va")).size() == 0) << std::endl;
    return 0;
}
