#pragma once

#include <string>

#include "../bow.h"

#include <httpi/html/html.h>
#include <httpi/displayer.h>

std::string PageGlobal(const std::string& content);
httpi::html::Html ClassifyResult(BoWClassifier& bow, const BowResult& bowr);
httpi::html::Html DisplayWeights(BoWClassifier& bow);
