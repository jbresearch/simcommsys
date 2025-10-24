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

#include "montecarlo.h"

#include "fsm.h"
#include "itfunc.h"
#include "randgen.h"
#include <limits>
#include <sstream>

namespace libcomm
{

using libbase::vector;

// worker processes

void
montecarlo::slave_getcode(void)
{
    system.reset();
    // Receive system as a string
    std::string systemstring;
    cluster.receive(systemstring);
    // Create system object from serialization
    std::istringstream is(systemstring);
    is >> system;
    // Compute its digest
    is.seekg(0);
    sysdigest.process(is);
    // Tell the user what we've done
    std::cerr << "Date: " << libbase::timer::date() << std::endl;
    std::cerr << system->description() << std::endl;
    std::cerr << "Digest: " << std::string(sysdigest) << std::endl;
}

void
montecarlo::slave_getparameters(void)
{
    std::cerr << "Date: " << libbase::timer::date() << std::endl;

    seed_experiment();
    libbase::vector<double> params;
    cluster.receive(params);
    system->set_parameters(params);

    std::cerr << "Simulating system at parameters = ";
    params.serialize(std::cerr, ", ");
    std::cerr << std::endl;
}

void
montecarlo::slave_work(void)
{
    // Initialise running values
    system->reset();

    // Iterate for 500ms, which is a good compromise between efficiency and
    // usability
    libbase::walltimer tslave("montecarlo_slave");
    while (tslave.elapsed() < 0.5) {
        sampleandaccumulate();
    }
    tslave.stop(); // to avoid expiry

    // Send system digest and current parameter back to master
    cluster.send(sysdigest);
    cluster.send(system->get_parameters());

    // Send accumulated results back to master
    libbase::vector<double> state;
    system->get_state(state);
    cluster.send(system->get_samplecount());
    cluster.send(state);

    // print something to inform the user of our progress
    vector<double> result, errormargin;
    compute_estimate(result, errormargin);
    display(result, errormargin);
}

// helper functions

std::string
montecarlo::get_systemstring()
{
    std::ostringstream os;
    os << system;
    std::string systemstring = os.str();
    return systemstring;
}

/*! \brief Seed the random generators in the experiment
 *
 * Use the stored seed to initialize a PRNG for seeding the embedded system.
 */
void
montecarlo::seed_experiment()
{
    libbase::randgen prng;
    prng.seed(seed);
    system->seedfrom(prng);
    std::cerr << "Seed: " << seed << std::endl;
}

// overrideable user-interface functions

/*!
 * \brief Default progress display routine.
 *
 * \note Display updates are rate-limited
 */
void
montecarlo::display(const libbase::vector<double>& result,
                    const libbase::vector<double>& errormargin) const
{
    if (tupdate.elapsed() > 0.5) {
        const std::streamsize prec = std::clog.precision(3);
        std::clog << "Timer: " << t << ", ";
        if (cluster.isenabled()) {
            std::clog << cluster.getnumslaves() << " clients, ";
        } else {
            std::clog << "local, ";
        }
        std::clog << cluster.getcputime() / t.elapsed() << "× usage, ";
        std::clog << "pass " << system->get_samplecount() << "." << std::endl;
        std::clog << "System parameters: ";
        libbase::vector<double> params = system->get_parameters();
        params.serialize(std::clog, ", ");
        std::clog << std::endl;
        std::clog << "Results:" << std::endl;
        system->prettyprint_results(std::clog, result, errormargin);
        std::clog << "Press 'q' to interrupt." << std::endl;
        std::clog.precision(prec);
        tupdate.start();
    }
}

// main process

/*!
 * \brief Determine overall estimate from accumulated results
 * \param[out] result      Vector containing the set of estimates
 * \param[out] errormargin Corresponding margin of error (radius of confidence
 * interval)
 */
void
montecarlo::compute_estimate(vector<double>& result,
                          vector<double>& errormargin) const
{
    const double cfactor = libbase::Qinv((1.0 - confidence) / 2.0);
    // determine a new estimate
    system->estimate(result, errormargin);
    assert(result.size() == errormargin.size());
    // determine confidence interval from standard error
    errormargin *= cfactor;
}

/*!
 * \brief Initialize given slave
 * \param   s              Slave to be initialized
 * \param   systemstring   Serialized system description
 *
 * Initialize given slave by sending the system being simulated and the
 * current simulation parameter.
 */
void
montecarlo::initslave(std::shared_ptr<libbase::socket> s,
                      std::string systemstring)
{
    try {
        cluster.call(s, "slave_getcode");
        cluster.send(s, systemstring);
        cluster.call(s, "slave_getparameters");
        cluster.send(s, system->get_parameters());
        libbase::trace << "DEBUG (estimate): Slave (" << s
                       << ") initialized ok." << std::endl;
    } catch (std::runtime_error& e) {
        std::cerr << "Runtime exception: " << e.what() << std::endl;
    }
}

/*!
 * \brief Initialize any new slaves
 * \param   systemstring   Serialized system description
 *
 * If there are any slaves in the NEW state, initialize them by sending the
 * system being simulated and the current simulation parameters.
 */
void
montecarlo::initnewslaves(std::string systemstring)
{
    while (std::shared_ptr<libbase::socket> s = cluster.find_new_slave()) {
        libbase::trace << "DEBUG (estimate): New slave found (" << s
                       << "), initializing." << std::endl;
        initslave(s, systemstring);
    }
}

/*!
 * \brief Get idle slaves to work if we're not yet done
 * \param   converged  True if results have already converged
 *
 * If there are any slaves in the IDLE state, ask them to start working. We ask
 * *all* IDLE slaves to work, as long as the results have not yet converged.
 * Therefore, this happens when the target accuracy is not yet reached or if the
 * number of samples gathered is not yet enough. This necessarily causes
 * extraneous results to be computed; these will then be discarded during the
 * next turn. This method avoids the master hanging up waiting for results from
 * slaves that will never come (happens if the machine is locked up but the
 * TCP/IP stack is still running).
 */
void
montecarlo::workidleslaves(bool converged)
{
    for (std::shared_ptr<libbase::socket> s;
         (!converged) && (s = cluster.find_idle_slave());) {
        try {
            libbase::trace << "DEBUG (estimate): Idle slave found (" << s
                           << "), assigning work." << std::endl;
            cluster.call(s, "slave_work");
            libbase::trace << "DEBUG (estimate): Slave (" << s
                           << ") work assigned ok." << std::endl;
        } catch (std::runtime_error& e) {
            std::cerr << "Runtime exception: " << e.what() << std::endl;
        }
    }
}

/*!
 * \brief Read and accumulate results from any pending slaves
 * \return  True if any new results have been added, false otherwise
 *
 * If there are any slaves in the EVENT_PENDING state, read their results.
 * Values returned are accumulated into the running totals.
 *
 * If any slave returns a result that does not correspond to the same system
 * or parameter that are now being simulated, this is discarded and the slave
 * is marked as 'new'.
 */
bool
montecarlo::readpendingslaves()
{
    bool results_available = false;
    while (std::shared_ptr<libbase::socket> s = cluster.find_pending_slave()) {
        try {
            libbase::trace << "DEBUG (estimate): Pending event from slave ("
                           << s << "), trying to read." << std::endl;
            // get digest and parameter for simulated system
            std::string simdigest;
            libbase::vector<double> simparameters;
            cluster.receive(s, simdigest);
            cluster.receive(s, simparameters);
            // set up space for results that need to be returned
            uint64_t estsamplecount = 0;
            vector<double> eststate;
            // get results
            cluster.receive(s, estsamplecount);
            cluster.receive(s, eststate);
            // check that results correspond to system under simulation
            if (std::string(sysdigest) != simdigest ||
                simparameters.isnotequalto(system->get_parameters())) {
                libbase::trace
                    << "DEBUG (estimate): Slave returned invalid results (" << s
                    << "), re-initializing." << std::endl;
                cluster.resetslave(s);
                continue;
            }
            // accumulate
            system->accumulate_state(estsamplecount, eststate);
            // update usage information and return flag
            cluster.updatecputime(s);
            results_available = true;
            libbase::trace << "DEBUG (estimate): Read from slave (" << s
                           << ") succeeded." << std::endl;
        } catch (std::runtime_error& e) {
            std::cerr << "Runtime exception: " << e.what() << std::endl;
        }
    }
    return results_available;
}

// Main process

/*!
 * \brief Simulate the system until convergence to given accuracy & confidence,
 * and return estimated results
 * \param[out] result      Vector of results
 * \param[out] errormargin Vector of corresponding margin of error (radius of
 * confidence interval)
 */
void
montecarlo::estimate(vector<double>& result, vector<double>& errormargin)
{
    t.start();

    // Initialise running values
    system->reset();
    // create string representation of system
    std::string systemstring = get_systemstring();
    // compute its digest
    std::istringstream is(systemstring);
    sysdigest.process(is);

    // Initialize results-writing system (if we're using it)
    // Note: this method is used in cases (e.g. quicksimulation) where we do not
    // have a results file, so the factory is never called.
    if (results_file != nullptr && results_file->isinitialized()) {
        results_file->setupfile();
    }

    // Set up for master-slave system (if necessary)
    // and seed the experiment
    if (cluster.isenabled()) {
        cluster.resetslaves();
        cluster.resetcputime();
    } else {
        seed_experiment();
    }

    // Repeat the experiment until all the following are true:
    // 1) We have the accuracy we need
    // 2) We have enough samples for the accuracy to be meaningful
    // An interrupt from the user overrides everything...
    bool converged = false;
    while (!converged) {
        bool results_available = false;
        // repeat the experiment
        if (cluster.isenabled()) {
            // first initialize any new slaves
            initnewslaves(systemstring);
            // get idle slaves to work if we're not yet done
            workidleslaves(converged);
            // wait for results, but not indefinitely - this allows user to
            // break
            cluster.waitforevent(true, 0.5);
            // accumulate results from any pending slaves
            results_available = readpendingslaves();
        } else {
            sampleandaccumulate();
            results_available = true;
        }
        // if we did get any results, update the statistics
        if (results_available) {
            compute_estimate(result, errormargin);
            // if we have done enough samples, check accuracy reached
            if (system->get_samplecount() >= uint64_t(min_samples)) {
                switch (mode) {
                case mode_relative_error: {
                    // determine error margin as a fraction of result mean
                    const vector<double> result_acc = errormargin / result;
                    // check if this is less than threshold
                    if (result_acc.max() <= threshold) {
                        converged = true;
                    }
                    break;
                }
                case mode_absolute_error: {
                    // check if error margin is less than threshold
                    if (errormargin.max() <= threshold) {
                        converged = true;
                    }
                    break;
                }
                case mode_accumulated_result: {
                    // TODO: re-implement or remove, now that get_samplecount(i)
                    // no longer exists

                    // // determine the absolute accumulated result
                    // vector<double> result_acc = result;
                    // for (int i = 0; i < result_acc.size(); i++) {
                    //     result_acc(i) *= system->get_samplecount(i);
                    // }
                    // // check if this is more than threshold
                    // if (result_acc.min() >= threshold) {
                    //     converged = true;
                    // }
                    break;
                }
                default:
                    failwith("Convergence mode not supported.");
                    break;
                }
            }
            // print something to inform the user of our progress
            display(result, errormargin);
            // write interim results
            // Note: this method is used in cases (e.g. quicksimulation) where
            // we do not have a results file, so the factory is never called.
            if (results_file != nullptr && results_file->isinitialized()) {
                results_file->writeinterimresults(result, errormargin);
            }
        }

        // consider our work done if the user has interrupted the processing
        // (note: this overrides everything)
        if (interrupt()) {
            break;
        }
    }

    // write final results
    // Note: this method is used in cases (e.g. quicksimulation) where we do not
    // have a results file, so the factory is never called.
    if (results_file != nullptr && results_file->isinitialized()) {
        results_file->writefinalresults(result, errormargin, interrupt());
    }

    t.stop();
}

} // namespace libcomm
