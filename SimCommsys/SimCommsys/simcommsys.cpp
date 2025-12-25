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

#include "cputimer.h"
#include "experiment/binomial/commsys_simulator.h"
#include "masterslave.h"
#include "montecarlo.h"
#include "randgen.h"
#include "range.h"
#include "serializer_libcomm.h"

#include <boost/program_options.hpp>

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>

using std::cerr;
using std::cout;
using std::setprecision;
namespace po = boost::program_options;

class mymontecarlo : public libcomm::montecarlo
{
private:
    bool quiet;    //!< Flag to disable intermediate displays
    bool hard_int; //!< Flag indicating a hard interrupt (stop completely)
    bool soft_int; //!< Flag indicating a soft interrupt (skip to next point)
public:
    mymontecarlo(bool quiet) : quiet(quiet), hard_int(false), soft_int(false) {}
    /*! \brief Conditional progress display
     *
     * If the object was set up to be quiet, then no display occurs, otherwise
     * use the default.
     */
    void display(const libbase::vector<double>& result,
                 const libbase::vector<double>& errormargin) const
    {
        if (!quiet) {
            libcomm::montecarlo::display(result, errormargin);
        }
    }
    /*! \brief User-interrupt check (public to allow use by main program)
     * This function returns true if the user has requested a soft or hard
     * interrupt. As required by the interface, once it returns true, all
     * subsequent evaluations keep returning true again.
     * A soft interrupt can be checked for and reset; hard interrupts cannot.
     * Checks for user pressing 's' (soft), 'q' or Ctrl-C (hard).
     */
    bool interrupt()
    {
        if (hard_int || soft_int) {
            return true;
        }

        if (libbase::interrupted()) {
            hard_int = true;
        } else if (libbase::keypressed() > 0) {
            const char k = libbase::readkey();
            hard_int = (k == 'q');
            soft_int = (k == 's');
        }

        return hard_int || soft_int;
    }
    /*! \brief Soft-interrupt check and reset
     * This should be called after interrupt(), and returns true if the
     * interrupt found was a soft interrupt. This method also resets the
     * soft interrupt condition when found.
     */
    bool interrupt_was_soft()
    {
        const bool result = soft_int;
        soft_int = false;
        return result;
    }
};

std::shared_ptr<libcomm::experiment>
createsystem(const std::string& fname)
{
    const libcomm::serializer_libcomm my_serializer_libcomm;
    // load system from string representation
    std::shared_ptr<libcomm::experiment> system;
    std::ifstream file(fname.c_str(),
                       std::ios_base::in | std::ios_base::binary);
    file >> system >> libbase::verifycomplete;
    return system;
}

/*!
 * \brief   Simulation of Communication Systems
 * \author  Johann Briffa
 */

int
main(int argc, char* argv[])
{
    libbase::cputimer tmain("Main timer");

    // Set up user parameters
    po::options_description desc("Allowed options");
    desc.add_options()("help", "print this help message");
    desc.add_options()(
        "quiet,q", po::bool_switch(), "suppress all output except benchmark");
    desc.add_options()(
        "priority,p", po::value<int>()->default_value(10), "process priority");
    desc.add_options()("endpoint,e",
                       po::value<std::string>()->default_value("local"),
                       "- 'local', for local-computation model\n"
                       "- ':port', for server-mode, bound to given port\n"
                       "- 'hostname:port', for client-mode connection");
    desc.add_options()("system-file,i",
                       po::value<std::string>(),
                       "input file containing system description");
    desc.add_options()("results-file,o",
                       po::value<std::string>(),
                       "output file to hold results");
    desc.add_options()("param-range,r",
                       po::value<std::vector<libbase::range>>()->multitoken(),
                       "parameter ranges for simulation");
    desc.add_options()("floor-min",
                       po::value<double>(),
                       "stop simulation when at least one result converges "
                       "below this threshold");
    desc.add_options()(
        "floor-max",
        po::value<double>(),
        "stop simulation when all results converge below this threshold");
    desc.add_options()(
        "confidence",
        po::value<double>()->default_value(0.90),
        "confidence level for computing margin of error (e.g. 0.90 for 90%)");
    desc.add_options()("relative-error",
                       po::value<double>()->default_value(0.15),
                       "target error margin, as a fraction of result mean "
                       "(e.g. 0.15 for ±15%)");
    desc.add_options()(
        "absolute-error",
        po::value<double>(),
        "target error margin, as an absolute value (e.g. 0.1 for ±0.1); "
        "overrides relative-error if specified");
    desc.add_options()(
        "accumulated-result",
        po::value<double>(),
        "target accumulated result (i.e. result mean x sample count); "
        "overrides absolute and relative error if specified");
    desc.add_options()(
        "min-samples", po::value<uint64_t>(), "minimum number of samples");
    desc.add_options()(
        "max-samples", po::value<uint64_t>(), "maximum number of samples");
    desc.add_options()("seed,s",
                       po::value<uint32_t>(),
                       "system initialization seed (random if not stated)");
    desc.add_options()("output-format,f",
                       po::value<std::string>()->default_value("text"),
                       "output format; use text for regular human-readable "
                       "output, json for machine-readable JSON output.");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // Validate user parameters
    if (vm.count("help")) {
        cout << desc << std::endl;
        return 0;
    }

    // Create estimator object and initilize cluster
    mymontecarlo estimator(vm["quiet"].as<bool>());
    switch (estimator.enable(vm["endpoint"].as<std::string>(),
                             vm["quiet"].as<bool>(),
                             vm["priority"].as<int>())) {
    case libbase::masterslave::mode_slave:
        break;

    case libbase::masterslave::mode_local:
    case libbase::masterslave::mode_master:
        // If this is a server instance, check the remaining parameters
        if (vm.count("system-file") == 0 || vm.count("results-file") == 0 ||
            vm.count("param-range") == 0) {
            cout << desc << std::endl;
            return 0;
        }

        // main process
        {
            // Simulation system & parameters
            estimator.set_output_format(vm["output-format"].as<std::string>());
            estimator.set_resultsfile(vm["results-file"].as<std::string>());
            std::shared_ptr<libcomm::experiment> system =
                createsystem(vm["system-file"].as<std::string>());
            estimator.bind(system);
            libbase::vector<libbase::range> param_ranges =
                (libbase::vector<libbase::range>)vm["param-range"]
                    .as<std::vector<libbase::range>>();

            estimator.set_confidence(vm["confidence"].as<double>());

            if (vm.count("accumulated-result")) {
                estimator.set_accumulated_result(
                    vm["accumulated-result"].as<double>());
            } else if (vm.count("absolute-error")) {
                estimator.set_absolute_error(vm["absolute-error"].as<double>());
            } else {
                estimator.set_relative_error(vm["relative-error"].as<double>());
            }

            if (vm.count("min-samples")) {
                estimator.set_min_samples(vm["min-samples"].as<uint64_t>());
            }
            if (vm.count("max-samples")) {
                estimator.set_max_samples(vm["max-samples"].as<uint64_t>());
            }
            if (vm.count("seed")) {
                estimator.set_seed(vm["seed"].as<uint32_t>());
            }

            // Work out the following for every combination of parameters
            // required
            auto params_it = libbase::multi_range_iterator::begin(param_ranges);
            int params_count = 1;
            int params_total_count = params_it.count();
            for (;
                 params_it <= libbase::multi_range_iterator::end(param_ranges);
                 ++params_it, params_count++) {
                libbase::vector<double> params = *params_it;
                system->set_parameters(params);

                cerr << "[" << params_count << "/" << params_total_count
                     << "] Simulating system at parameters = ";
                params.serialize(cerr, ", ");
                libbase::vector<double> estimate, errormargin;
                estimator.estimate(estimate, errormargin);

                cerr << "Statistics: " << setprecision(4)
                     << estimator.get_samplecount() << " samples in "
                     << estimator.get_timer() << " - "
                     << estimator.get_samplecount() /
                            estimator.get_timer().elapsed()
                     << " samples/sec" << std::endl;

                // handle pre-mature breaks
                if (estimator.interrupt() && !estimator.interrupt_was_soft()) {
                    break;
                }

                if (vm.count("floor-min") &&
                    estimate.min() < vm["floor-min"].as<double>()) {
                    break;
                }

                if (vm.count("floor-max") &&
                    estimate.max() < vm["floor-max"].as<double>()) {
                    break;
                }
            }
        }
        break;
    }

    return 0;
}
