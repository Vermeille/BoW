#pragma once

#include <string>

#include "../bow.h"

#include <httpi/html/html.h>
#include <httpi/displayer.h>

std::string PageGlobal(const std::string& content);
std::string Classify(BoWClassifier& bow, const std::string&,
        const POSTValues& args);
httpi::html::Html DisplayWeights(BoWClassifier& bow);
