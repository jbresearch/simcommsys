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

#include "resultsfile_text.h"
#include "montecarlo.h"
#include "vector.h"
#include "version.h"
#include <cassert>
#include <fstream>

namespace libcomm
{

void
resultsfile_text::writeheader(std::ostream& sout) const
{
    assert(sout.good());
    assert(system != NULL);
    libbase::trace << "DEBUG (resultsfile_text): writing results header."
                   << std::endl;
    // Print information on the simulation being performed
    libbase::trace << "DEBUG (resultsfile_text): position before = "
                   << sout.tellp() << std::endl;
    sout << "#% " << system->description() << std::endl;
    sout << "#% Confidence Level: " << simulator->get_confidence_level()
         << std::endl;
    sout << "#% Convergence Mode: " << simulator->get_convergence_mode()
         << std::endl;
    sout << "#% Date: " << libbase::timer::date() << std::endl;
    sout << "#% Build: " << SIMCOMMSYS_BUILD << std::endl;
    sout << "#% Version: " << SIMCOMMSYS_VERSION << std::endl;
    sout << "#" << std::endl;

    /// Print results header
    // We must account for multiple params
    const int n_params = system->get_parameters().size();
    assertalways(n_params > 0);
    // if there is only one param, print old header with Par
    if (n_params == 1) {
        sout << "# Par";
    } else {
        sout << "# Par1";
    }
    // print rest of params after first one
    for (int i = 1; i < n_params; i++) {
        sout << "\t" << "Par" << i + 1;
    }
    // print header for each result and its tolerance
    for (int i = 0; i < system->result_count(); i++) {
        sout << "\t" << system->result_description(i) << "\t"
             << system->result_description(i) << "_Tol";
    }
    sout << "\tSamples\tCPUtime" << std::endl;
    libbase::trace << "DEBUG (resultsfile_text): position after = "
                   << sout.tellp() << std::endl;
}

/*! \brief Checks whether header has already been written and writes it if
 * not.
 *
 * The resultsfile_text::headerwritten member variable keeps track of
 * whether or not header has been written.
 *
 * \note This method also updates the write position so that the header is
 * not overwritten on the next write.
 */
void
resultsfile_text::writeheaderifneeded(std::fstream& file)
{
    if (!headerwritten) {
        writeheader(file);
        // update flag
        headerwritten = true;
        // update file-write position
        fileptr = file.tellp();
    }
}

void
resultsfile_text::writeresults(std::ostream& sout,
                               libbase::vector<double>& result,
                               libbase::vector<double>& errormargin) const
{
    assert(sout.good());
    if (simulator->get_samplecount() == 0) {
        return;
    }

    libbase::trace << "DEBUG (resultsfile_text): writing results." << std::endl;
    // Write current estimates to file
    libbase::trace << "DEBUG (resultsfile_text): position before = "
                   << sout.tellp() << std::endl;
    // print all the param values (cannot use serialize, to avoid final eol)
    const libbase::vector<double> params = system->get_parameters();
    assertalways(params.size() > 0);
    sout << params(0);
    for (int i = 1; i < params.size(); i++) {
        sout << '\t' << params(i);
    }
    // print results and their tolerances
    for (int i = 0; i < system->result_count(); i++) {
        sout << '\t' << result(i) << '\t' << errormargin(i);
    }

    sout << '\t' << simulator->get_samplecount();
    sout << '\t' << simulator->get_cluster().getcputime() << std::endl;
    libbase::trace << "DEBUG (resultsfile_text): position after = "
                   << sout.tellp() << std::endl;
}

void
resultsfile_text::writestate(std::ostream& sout) const
{
    assert(sout.good());
    if (simulator->get_samplecount() == 0) {
        return;
    }

    libbase::trace << "DEBUG (resultsfile_text): writing state." << std::endl;
    // Write accumulated values to file
    libbase::trace << "DEBUG (resultsfile_text): position before = "
                   << sout.tellp() << std::endl;
    libbase::vector<double> state;
    system->get_state(state);
    sout << "## System: " << simulator->get_sysdigest() << std::endl;
    sout << "## Parameters: " << system->get_parameters().size() << '\t';
    system->get_parameters().serialize(sout, "\t");
    sout << "## Samples: " << simulator->get_samplecount() << std::endl;
    sout << "## State: " << state.size() << '\t';
    state.serialize(sout, "\t");
    sout << std::flush;
    libbase::trace << "DEBUG (resultsfile_text): position after = "
                   << sout.tellp() << std::endl;
}

void
resultsfile_text::lookforstate(std::fstream& sin)
{
    assert(sin.good());
    // state variables to read
    std::string digest;
    libbase::vector<double> parameters;
    uint64_t samplecount = 0;
    libbase::vector<double> state;
    // read through entire file
    libbase::trace << "DEBUG (resultsfile_text): looking for state."
                   << std::endl;
    sin.seekg(0);
    while (!sin.eof()) {
        std::string s;
        getline(sin, s);

        if (s.substr(0, 11) == "## System: ") {
            digest = s.substr(11);
        } else if (s.substr(0, 15) == "## Parameters: ") {
            std::istringstream(s.substr(15)) >> parameters;
        } else if (s.substr(0, 12) == "## Samples: ") {
            std::istringstream(s.substr(12)) >> samplecount;
        } else if (s.substr(0, 10) == "## State: ") {
            std::istringstream(s.substr(10)) >> state;
        }
    }
    // reset file
    sin.clear();
    // check that results correspond to system under simulation
    if (digest == std::string(simulator->get_sysdigest()) &&
        parameters.isequalto(system->get_parameters())) {
        std::cerr << "NOTICE: Reloading state with " << samplecount
                  << " samples." << std::endl;
        system->accumulate_state(samplecount, state);
    }
}

/*! \brief Checks if file has been modified using resultsfile::wasmodified(), if
 * yes sets fileptr to EOF.
 *
 * This ensures that if file is modified by a user or external process while
 * we are handling it, we start appending to the end of the file instead of
 * risking a corrupted file.
 */
void
resultsfile_text::checkformodifications(std::fstream& file)
{
    if (wasmodified(file)) {
        std::cerr << "NOTICE: file modifications found - appending."
                  << std::endl;
        // set current write position to end-of-file
        file.seekp(0, std::ios_base::end);
        fileptr = file.tellp();
    } else {
        file.seekp(fileptr);
    }
}

/*! \brief Write current results and perhaps the state
 *
 * \note If the result being written is final rather than interim, the write
 * position is updated so that it is not overwritten. Otherwise it is not
 * updated.
 */
void
resultsfile_text::writeresultsandstate(std::fstream& file,
                                       libbase::vector<double>& result,
                                       libbase::vector<double>& errormargin,
                                       bool savestate,
                                       bool interim)
{
    checkformodifications(file);
    writeheaderifneeded(file);
    writeresults(file, result, errormargin);
    if (savestate)
        writestate(file);
    if (!interim) {
        // update position so we don't overwrite these results on next run
        fileptr = file.tellp();
        // truncate to remove extra content related to state
        this->truncate(fileptr);
    }
}

/*! \brief Set up the results file and look for a state
 *
 * If the file does not exist, a new one is created. Otherwise, the write
 * point is set to the end of file and a digest of the current file contents
 * is kept. A search for a saved state is also initiated by this method.
 */
void
resultsfile_text::setupfile()
{
    resultsfile::setupfile();
    // open file for input and output
    std::fstream file(this->get_fname().c_str());
    assertalways(file);
    // set write position at end
    file.seekp(0, std::ios_base::end);
    fileptr = file.tellp();
}

} // namespace libcomm
