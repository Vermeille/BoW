#pragma once

#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <future>
#include <memory>

#include "displayer.h"
#include "html.h"
#include "chart.h"

// Represent the arguments a job can take in order to define them.
// TODO: type validator
class Arg {
    std::string name_;  // formal name
    std::string type_;  // from now, it takes the possibles values of an HTML's input tag
    std::string desc_;  // short text describing the argument
  public:

    Arg(const std::string& name, const std::string& type, const std::string& desc);

    const std::string& name() const { return name_; }
    const std::string& type() const { return type_; }

    Html ArgToForm() const;
};

// Describe a job, generate the HTML for jobs.
class JobDesc {
    std::vector<Arg> args_;  // the job's prototype
    std::string name_;  // short name of the job shown in lists etc
    std::string url_;  // url on which this job will be mapped. Must start with /
    std::string desc_;  // text describing what the job does
    bool synchronous_;  // if synchronous, result will be given immediately. If not, queue a job.
    bool reentrant_;  // if not reentrant, only one running instance of the job is allowed
    std::function<Html(const std::vector<std::string>&, size_t job_id)> exec_;  // the actual function called
    std::vector<Chart> charts_;
  public:

    typedef std::function<Html(const std::vector<std::string>&, size_t job_id)> function_type;

    const std::vector<Arg>& args() const { return args_; }
    const std::string& name() const { return name_; }
    const std::string& url() const { return url_; }
    const std::string& description() const { return desc_; }
    bool IsSynchronous() const { return synchronous_; }
    function_type function() const { return exec_; }

    const std::vector<Chart>& charts() const { return charts_; }

    JobDesc() = default;
    JobDesc(const std::vector<Arg>& args, const std::string& name, const std::string& url,
            const std::string& desc, bool synchronous, bool reentrant,
            const function_type& fun, const std::vector<Chart>& charts = {});

    // return true and a vector of parameters if all the arguments are present in vs
    // return false and an error page if they're not
    std::tuple<bool, Html, std::vector<std::string>> ValidateParams(const POSTValues& vs);

    Html MakeForm() const;
    Html DisplayResult(const Html& res) const;
};

// Describe a running instance of a job
class JobStatus {
    std::chrono::system_clock::time_point start_;
    mutable std::future<Html> job_;
    std::vector<std::string> args_;
    mutable Html result_;
    mutable bool finished_;
    const JobDesc* const desc_;
    const size_t id_;
  public:

    size_t id() const { return id_; }
    std::string start_time() const {
        auto time = std::chrono::system_clock::to_time_t(start_);
        return std::ctime(&time);
    }

    const JobDesc* description() const { return desc_; }

    JobStatus(JobStatus&&) = default;
    JobStatus(const JobDesc* desc, const std::vector<std::string>& args, size_t id);

    Html result() const;

    bool IsFinished() const;
};

// pool keeping all the running instances, allowing to start a new job, check their statuses, etc
class RunningJobs {
    std::map<size_t, JobStatus> statuses;
    std::map<std::string, JobDesc> descriptors_;
    size_t next_id_ = 1;
  public:

    RunningJobs() = default;
    RunningJobs(RunningJobs&&) = default;

    void AddDescriptor(const JobDesc& jd) {
        descriptors_[jd.url()] = jd;
    }

    JobDesc* FindDescriptor(const std::string& url);
    Html RenderTableOfRunningJobs() const;
    Html RenderListOfDescriptors() const;
    JobStatus* FindJobWithId(size_t id);
    Html Exec(const std::string& url, const POSTValues& vs);
};

void RegisterJob(const JobDesc& jd);  // maps the job to its url etc
