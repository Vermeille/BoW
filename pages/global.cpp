#include "pages.h"

#include <httpi/html/html.h>

std::string PageGlobal(const std::string& content) {
    using namespace httpi::html;
    return (httpi::html::Html() <<
        "<!DOCTYPE html>"
        "<html>"
           "<head>"
                R"(<meta charset="utf-8">)"
                R"(<meta http-equiv="X-UA-Compatible" content="IE=edge">)"
                R"(<meta name="viewport" content="width=device-width, initial-scale=1">)"
                R"(<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap.min.css">)"
                R"(<link rel="stylesheet" href="//cdn.jsdelivr.net/chartist.js/latest/chartist.min.css">)"
                R"(<script src="//cdn.jsdelivr.net/chartist.js/latest/chartist.min.js"></script>)"
            "</head>"
            "<body lang=\"en\">"
                "<div class=\"container\">"
                    "<div class=\"col-md-9\">" <<
                        content <<
                    "</div>"
                    "<div class=\"col-md-3\">" <<
                        H2() << "Go to" << Close() <<
                        Ul() <<
                            Li() <<
                                A().Attr("href", "/jobs") <<
                                    "Jobs" <<
                                Close() <<
                            Close() <<
                            Li() <<
                                A().Attr("href", "/prediction") <<
                                    "Prediction" <<
                                Close() <<
                            Close() <<
                            Li() <<
                                A().Attr("href", "/dataset") <<
                                    "Dataset" <<
                                Close() <<
                            Close() <<
                            Li() <<
                                A().Attr("href", "/model") <<
                                    "Model" <<
                                Close() <<
                            Close() <<
                        Close() <<
                    "</div>"
                "</div>"
            "</body>"
        "</html>").Get();
}

