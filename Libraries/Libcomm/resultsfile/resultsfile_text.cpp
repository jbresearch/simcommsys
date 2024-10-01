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

    // Print results header
    sout << "# Par";
    for (int i = 0; i < system->count(); i++) {
        sout << "\t" << system->result_description(i) << "\tTol";
    }
    sout << "\tSamples\tCPUtime" << std::endl;
    libbase::trace << "DEBUG (resultsfile_text): position after = "
                   << sout.tellp() << std::endl;
}

/*! \brief If this is the first time, write the header
 * \note This method also updates the write position so that the header is not
 * overwritten on the next write.
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
    sout << system->get_parameter();

    for (int i = 0; i < system->count(); i++) {
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
    sout << "## Parameter: " << system->get_parameter() << std::endl;
    sout << "## Samples: " << simulator->get_samplecount() << std::endl;
    sout << "## State: " << state.size() << '\t';
    state.serialize(sout, '\t');
    sout << std::flush;
    libbase::trace << "DEBUG (resultsfile_text): position after = "
                   << sout.tellp() << std::endl;
}

void
resultsfile_text::lookforstate(std::istream& sin)
{
    assert(sin.good());
    // state variables to read
    std::string digest;
    double parameter = 0;
    libbase::int64u samplecount = 0;
    libbase::vector<double> state;
    // read through entire file
    libbase::trace << "DEBUG (resultsfile_text): looking for state."
                   << std::endl;
    sin.seekg(0);
    while (!sin.eof()) {
        std::string s;
        getline(sin, s);

        if (s.substr(0, 10) == "## System:") {
            digest = s.substr(10);
        } else if (s.substr(0, 13) == "## Parameter:") {
            std::istringstream(s.substr(13)) >> parameter;
        } else if (s.substr(0, 11) == "## Samples:") {
            std::istringstream(s.substr(11)) >> samplecount;
        } else if (s.substr(0, 9) == "## State:") {
            std::istringstream is(s.substr(9));
            is >> state;
        }
    }
    // reset file
    sin.clear();
    // check that results correspond to system under simulation
    if (digest == std::string(simulator->get_sysdigest()) &&
        parameter == system->get_parameter()) {
        std::cerr << "NOTICE: Reloading state with " << samplecount
                  << " samples." << std::endl;
        system->accumulate_state(samplecount, state);
    }
}

void
resultsfile_text::checkformodifications(std::fstream& file)
{
    if (wasmodified(file)) {
        file.seekp(fileptr);
    } else {
        std::cerr << "NOTICE: file modifications found - appending."
                  << std::endl;
        // set current write position to end-of-file
        file.seekp(0, std::ios_base::end);
        fileptr = file.tellp();
    }
}

/*! \brief Write current results and perhaps the state
 * \note If the state being written is final rather than interim, the write
 * position is updated so that it is not overwritten. Otherwise it is not
 * updated.
 */
void
resultsfile_text::writeresults(std::fstream& file,
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
    if (!interim)
        // update write-position
        fileptr = file.tellp();
}

/*! \brief Set up the results file and look for a state
 * If the file does not exist, a new one is created. Otherwise, the write
 * point is set to the end of file and a digest of the current file contents
 * is kept. A search for a saved state is also initiated by this method.
 *
 * \note The current simulation must be already set up at this point, so that
 * a valid comparison can be made.
 */
void
resultsfile_text::setupfile()
{
    resultsfile::setupfile();
    // open file for input and output
    std::fstream file(fname.c_str());
    assertalways(file);
    // set write position at end
    file.seekp(0, std::ios_base::end);
    fileptr = file.tellp();
}

} // namespace libcomm