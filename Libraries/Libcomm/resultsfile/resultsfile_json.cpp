/*!
 * \file
 *
 * Copyright (c) 2024 Mark Mizzi
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "resultsfile_json.h"
#include "assertalways.h"
#include "config.h"
#include "montecarlo.h"
#include "timer.h"
#include "version.h"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace libcomm
{

inline bool
isempty(std::fstream& f)
{
    std::streampos offset = f.tellg();
    f.seekg(0, std::ios_base::end);
    bool isempty = f.peek() == std::fstream::traits_type::eof();
    f.seekg(offset);
    return isempty;
}

json
resultsfile_json::readjson(std::fstream& sout) const
{
    json data;
    if (!isempty(sout)) {
        // file is not empty, we need to update contents of it.
        sout.seekg(0); // ensure we are reading from start of file.
        try {
            data = json::parse(sout);
        } catch (json::parse_error& e) {
            std::stringstream err;
            err << "JSON parse Error when reading results file " << get_fname()
                << " at byte " << e.byte;
            failwith(err.str());
        }
    }
    return data;
}

void
resultsfile_json::writejson(std::fstream& sout, const json& data) const
{
    // set write pos to start of file
    sout.seekp(0);
    // truncate contents of file, so we are writing to an empty file.
    this->truncate(sout.tellp());
    // get dump of data and write it to the file.
    std::string dump = data.dump();
    sout.write(dump.c_str(), dump.size());
    // set sout to point to end-of-file.
    sout.seekp(0, std::ios_base::end);
}

void
resultsfile_json::lookforstate(std::fstream& sin)
{
    assert(sin.good());

    // state variables to read
    std::string digest;
    double parameter = 0;
    libbase::int64u samplecount = 0;
    libbase::vector<double> state;
    // read through entire file
    libbase::trace << "DEBUG (resultsfile_json): looking for state."
                   << std::endl;

    json data = this->readjson(sin);

    if (data.contains("state")) {
        json& state_json = data["state"];
        if (state_json.contains("System"))
            digest = state_json["System"];
        if (state_json.contains("Parameter"))
            parameter = state_json["Parameter"];
        if (state_json.contains("Samples"))
            samplecount = state_json["Samples"];
        if (state_json.contains("State"))
            state = state_json["State"];
    }

    // check that results correspond to system under simulation
    if (digest == std::string(simulator->get_sysdigest()) &&
        parameter == system->get_parameter()) {
        std::cerr << "NOTICE: Reloading state with " << samplecount
                  << " samples." << std::endl;
        system->accumulate_state(samplecount, state);
    }
}

void
resultsfile_json::writeresults(std::fstream& sout,
                               libbase::vector<double>& result,
                               libbase::vector<double>& errormargin) const
{
    libbase::trace << "DEBUG (resultsfile_json): writing results." << std::endl;

    json data = this->readjson(sout);
    // results data is stored in the format
    /* {
     *      "results": {
     *          "param1": {
     *              "resname1": {
     *                  "value": ...,
     *                  "errormargin": ...
     *              },
     *              ...,
     *              "Samples": ...,
     *              "CPUtime": ...
     *          },
     *          ...
     *      },
     *      ...
     * }
     */
    // initialize data["results"]["paramN"] to {} and store a handy reference to
    // it.
    json& param_results =
        data["results"][std::to_string(system->get_parameter())] = json();
    for (int i = 0; i < system->count(); i++) {
        param_results[system->result_description(i)] = {
            {"value", result(i)}, {"errormargin", errormargin(i)}};
    }
    param_results["Samples"] = simulator->get_samplecount();
    param_results["CPUtime"] = simulator->get_cluster().getcputime();

    this->writejson(sout, data);
}

void
resultsfile_json::writestate(std::fstream& sout) const
{
    assert(sout.good());
    if (simulator->get_samplecount() == 0) {
        return;
    }

    libbase::trace << "DEBUG (resultsfile_json): writing state." << std::endl;

    libbase::vector<double> state;
    system->get_state(state);

    json data = this->readjson(sout);

    data["state"] = {{"System", simulator->get_sysdigest()},
                     {"Parameter", system->get_parameter()},
                     {"Samples", simulator->get_samplecount()},
                     {"State", (std::vector<double>)state}};

    this->writejson(sout, data);
}

void
resultsfile_json::writemetadata(std::fstream& sout) const
{
    assert(sout.good());
    assert(system != nullptr);
    libbase::trace << "DEBUG (resultsfile_json): writing results metadata."
                   << std::endl;

    json data = this->readjson(sout);

    data["System"] = system->description();
    data["Confidence Level"] = simulator->get_confidence_level();
    data["Convergence Mode"] = simulator->get_convergence_mode();
    data["Date"] = libbase::timer::date();
    data["Build"] = SIMCOMMSYS_BUILD;
    data["Version"] = SIMCOMMSYS_VERSION;

    this->writejson(sout, data);
}

void
resultsfile_json::writemetadataifneeded(std::fstream& sout) const
{
    if (isempty(sout)) {
        // file is empty, we are opening it for the first time.
        writemetadata(sout);
    }
}

} // namespace libcomm