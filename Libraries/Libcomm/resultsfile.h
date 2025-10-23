/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
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

#ifndef __resultsfile_h
#define __resultsfile_h

#include "config.h"

#include "experiment.h"
#include "sha.h"
#include "walltimer.h"
#include <iostream>

namespace libcomm
{

class montecarlo;

/*!
 * \brief   Results File Handler.
 * \author  Johann Briffa, Mark Mizzi
 *
 * This class encapsulates the process of writing results to a file. It opens
 * and closes the file for every write, ensuring that written results are
 * flushed, and also detecting external modifications of the file.
 *
 * The handler keeps track of the file digest between writes, checking for
 * any external changes. In such cases, the file is considered 'modified'.
 * External modifications are handled by subclasses which can choose whether or
 * not they are allowed; modifications can be detected within subclasses using
 * resultsfile::wasmodified() which checks the file against
 * resultsfile::filedigest.
 *
 * The handler also allows 'interim' result writing. In this case, the result
 * is written together with the simulation state. This allows the user to
 * continue an aborted simulation (due to simulator or machine crash, for
 * example). When a file is initialized, a search for the last-save simulator
 * state is performed. If this matches the current system at the current
 * parameter, this state needs to be loaded.
 *
 * \note The handler does not specify the format for writing results or state.
 * Instead, these functions are performed by the
 * resultsfile::writeresultsandstate() pure virtual method.
 *
 * Classes implementing this interface need to:
 * 1) provide implementations for virtual methods (writing results and state,
 * looking for a state)
 */

class resultsfile
{
protected:
    /*! \name Backpointers */
    montecarlo* simulator;
    experiment* system;
    // @}

private:
    /*! \name User-specified parameters */
    std::string fname; //!< Filename for associated results file
    // @}
    /*! \name Internal variables */
    bool filesetup;       //!< Flag to indicate that the results file was set up
    sha filedigest;       //!< Digest of file as at last update
    libbase::walltimer t; //!< Timer to keep track of running estimate
                          // @}

protected:
    const std::string& get_fname() const { return fname; }

    /*! \name Results file helper functions */
    void update_digest(std::fstream& file);
    void truncate(std::streampos length) const;
    bool wasmodified(std::fstream& file);
    virtual void lookforstate(std::fstream& sin) = 0;
    /*! \brief Write results and possibly state to output file.
     *
     * \note Implementing classes should always set \ref std::fstream to point
     * to the end of the data they are writing, as anything after will be
     * truncated.
     */
    virtual void writeresultsandstate(std::fstream& file,
                                      libbase::vector<double>& result,
                                      libbase::vector<double>& errormargin,
                                      bool savestate,
                                      bool interim) = 0;
    // @}
public:
    /*! \name Constructor/destructor */
    // Constructor/destructor
    resultsfile(montecarlo& simulator)
        : simulator(&simulator), filesetup(false), t("resultsfile", false)
    {
    }
    virtual ~resultsfile() { assert(!t.isrunning()); }
    // @}

    /*! \name File handling interface */
    /*! \brief Provide filename
     *
     * After this, the results handling interface methods can be used.
     */
    void init(const std::string& fname);
    /*! \brief Check whether the handler has been initialized
     *
     * Indicates whether the results handling interface methods can be used.
     */
    bool isinitialized() const { return !fname.empty(); }
    // @}

    /*! \name Results handling interface */
    virtual void setupfile();

    /*! \brief Write current results and state
     *
     * This method can be called as many times as required; usually this is
     * called after every update. It may be wise for implementing subclasses to
     * limit file writes to occur no more often than a certain frequency.
     */
    void writeinterimresults(libbase::vector<double>& result,
                             libbase::vector<double>& errormargin);

    /*! \brief Write final results and state
     *
     * This method is called when the final result is reached. A file write is
     * guaranteed to occur. If requested, the final state is also written.
     */
    void writefinalresults(libbase::vector<double>& result,
                           libbase::vector<double>& errormargin,
                           bool savestate = false);

    void set_system(experiment& system) { this->system = &system; }
    // @}
};

} // namespace libcomm

#endif
