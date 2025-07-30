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

#include "assertalways.h"
#include "montecarlo.h"
#include "serializer_libcomm.h"
#include "timer.h"
#include "vector.h"
#include "version.h"

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace po = boost::program_options;

/** \brief Subclass of montecarlo which stops simulation if simulation time has
 * exceeded timeout.
 */
class montecarlo_timeout : public libcomm::montecarlo
{
protected:
    bool interrupt()
    {
        return montecarlo::interrupt() || get_timer().elapsed() > timeout;
    }

public:
    double timeout;

    montecarlo_timeout(double timeout) : timeout(timeout) {}
};

/** \brief Subclass of montecarlo which stops simulation if samples taken
 * exceeds a maximum threshold.
 * \author Mark Mizzi
 */
class montecarlo_max_samples : public libcomm::montecarlo
{
protected:
    bool interrupt()
    {
        return montecarlo::interrupt() || get_samplecount() >= max_samples;
    }

public:
    unsigned max_samples;

    montecarlo_max_samples(unsigned max_samples) : max_samples(max_samples) {}
};

/*!
 * \brief   Quick Simulation
 * \author  Johann Briffa
 *
 * This program implements a quick simulation for a given system; this is
 * useful to benchmark the speed of the decoder and to obtain a quick estimate
 * for the performance of a code under given conditions.
 */

int
main(int argc, char* argv[])
{
    using std::cout;
    using std::setprecision;

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
    desc.add_options()("time,t",
                       po::value<double>()->default_value(60),
                       "benchmark duration in seconds");
    desc.add_options()(
        "num-samples,n",
        po::value<unsigned>()->default_value(static_cast<unsigned>(0)),
        "maximum number of samples taken in the simulation. Ignored if value "
        "given is 0 or no value is given. Overrides timeout if a +ve value is "
        "specified.");
    desc.add_options()("parameter,r",
                       po::value<std::vector<double>>()->multitoken(),
                       "simulation parameters (e.g. SNR)");
    desc.add_options()("system-file,i",
                       po::value<std::string>(),
                       "file containing system description");
    desc.add_options()("seed,s",
                       po::value<libbase::int32u>(),
                       "system initialization seed (random if not stated)");
    desc.add_options()(
        "confidence",
        po::value<double>()->default_value(0.999),
        "confidence level for computing margin of error (e.g. 0.90 for 90%)");
    desc.add_options()("relative-error",
                       po::value<double>()->default_value(0.001),
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
        "min-samples", po::value<int>(), "minimum number of samples");
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
        return 1;
    }

    // Create estimator object
    std::unique_ptr<libcomm::montecarlo> estimator;
    unsigned num_samples = vm["num-samples"].as<unsigned>();
    if (num_samples > 0) {
        estimator = std::make_unique<montecarlo_max_samples>(num_samples);
    } else {
        estimator =
            std::make_unique<montecarlo_timeout>(vm["time"].as<double>());
    }

    // Initilize cluster
    switch (estimator->enable(vm["endpoint"].as<std::string>(),
                              vm["quiet"].as<bool>(),
                              vm["priority"].as<int>())) {
    case libbase::masterslave::mode_slave:
        break;

    case libbase::masterslave::mode_local:
    case libbase::masterslave::mode_master: {
        // If this is a server instance, check the remaining parameters
        if (vm.count("system-file") == 0 || vm.count("parameter") == 0) {
            cout << desc << std::endl;
            return 0;
        }
        // Set up the estimator
        std::shared_ptr<libcomm::experiment> system;
        system = libcomm::loadfromfile<libcomm::experiment>(
            vm["system-file"].as<std::string>());
        estimator->bind(system);
        estimator->set_confidence(vm["confidence"].as<double>());

        if (vm.count("accumulated-result")) {
            estimator->set_accumulated_result(
                vm["accumulated-result"].as<double>());
        } else if (vm.count("absolute-error")) {
            estimator->set_absolute_error(vm["absolute-error"].as<double>());
        } else {
            estimator->set_relative_error(vm["relative-error"].as<double>());
        }

        if (vm.count("min-samples")) {
            estimator->set_min_samples(vm["min-samples"].as<int>());
        }
        if (vm.count("seed")) {
            estimator->set_seed(vm["seed"].as<libbase::int32u>());
        }

        // Work out at the SNR value required
        system->set_parameters(
            (libbase::vector<double>)vm["parameter"].as<std::vector<double>>());

        // Print some debug information
        libbase::trace << system->description() << std::endl;

        // Perform the simulation
        libbase::vector<double> estimate, errormargin;
        estimator->estimate(estimate, errormargin);
        const libbase::int64u samples = estimator->get_samplecount();

        if (!vm["quiet"].as<bool>()) {
            // Create a count of each result produced by the system.
            // For example if the system produces 4 results with the description
            // "t_decode_iter", we set result_descr_count["t_decode_iter"] = 4
            // Used when labelling results in the final output
            std::map<std::string, int> result_descr_count;
            for (int j = 0; j < system->count(); j++) {
                if (result_descr_count.count(system->result_description(j))) {
                    ++result_descr_count[system->result_description(j)];
                } else {
                    result_descr_count[system->result_description(j)] = 0;
                }
            }

            std::string output_format = vm["output-format"].as<std::string>();
            if (output_format == "text") {
                // Write some information on the code
                cout << std::endl << std::endl;
                cout << "System Used:" << std::endl;
                cout << "~~~~~~~~~~~~" << std::endl;
                cout << system->description() << std::endl;
                // cout << "Rate: " << system-> << std::endl;
                cout << "Confidence Level: "
                     << estimator->get_confidence_level() << std::endl;
                cout << "Convergence Mode: "
                     << estimator->get_convergence_mode() << std::endl;
                cout << "Date: " << libbase::timer::date() << std::endl;

                cout << "Simulating system at parameters = ";
                libbase::vector<double> params = system->get_parameters();
                params.serialize(cout, ", ");
                cout << std::endl;

                // Print results (for confirming accuracy)
                cout << std::endl;
                cout << "Results:" << std::endl;
                cout << "~~~~~~~~" << std::endl;

                // keep track of how many results with a particular name we have
                // seen so far
                std::map<std::string, int> result_descr_curr_count;
                for (int j = 0; j < system->count(); j++) {
                    // update result count
                    ++result_descr_curr_count[system->result_description(j)];

                    cout << system->result_description(j);
                    if (result_descr_count[system->result_description(j)] > 1) {
                        cout << result_descr_curr_count
                                [system->result_description(j)];
                    }
                    cout << '\t';
                    cout << setprecision(6) << estimate(j);
                    cout << "\t[±" << setprecision(3)
                         << fabs(100 * errormargin(j) / estimate(j)) << "%]";
                    cout << std::endl;
                }

                // Output timing statistics
                cout << std::endl;
                cout << "Build: " << SIMCOMMSYS_BUILD << std::endl;
                cout << "Version: " << SIMCOMMSYS_VERSION << std::endl;
                cout << "Statistics: " << samples << " samples in "
                     << estimator->get_timer() << "." << std::endl;

                // Output overall benchmark
                cout << "Simulation Speed: " << setprecision(4)
                     << samples / estimator->get_timer().elapsed()
                     << " samples/sec" << std::endl;

            } else if (output_format == "json") {
                nlohmann::basic_json output_json = {
                    {"System", system->description()},
                    {"Confidence Level", estimator->get_confidence_level()},
                    {"Convergence Mode", estimator->get_convergence_mode()},
                    {"Date", libbase::timer::date()},
                    {"System Parameters",
                     (std::vector<double>)system->get_parameters()},
                    {"Build", SIMCOMMSYS_BUILD},
                    {"Version", SIMCOMMSYS_VERSION},
                    {"Samples", samples},
                    {"Time", estimator->get_timer().elapsed()},
                    {"Simulation Speed",
                     samples / estimator->get_timer().elapsed()}};

                // keep track of how many results with a particular name we have
                // seen so far
                std::map<std::string, int> result_descr_curr_count;
                for (int j = 0; j < system->count(); j++) {
                    // update result count
                    ++result_descr_curr_count[system->result_description(j)];

                    double errmargin = fabs(100 * errormargin(j) / estimate(j));

                    std::string result_label = system->result_description(j);
                    if (result_descr_curr_count[system->result_description(j)] >
                        1) {
                        result_label += std::to_string(
                            result_descr_curr_count[system->result_description(
                                j)]);
                    }
                    output_json[result_label] = {{"Value", estimate(j)}};
                    if (std::isnan(errmargin))
                        output_json[result_label]["Tolerance"] = "NaN";
                    else
                        output_json[result_label]["Tolerance"] = errmargin;
                }

                cout << output_json.dump(3) << std::endl;
            } else {
                std::string error_msg(
                    "Invalid output format " + output_format +
                    " specified; accepted values are text or json.");
                failwith(error_msg.c_str());
            }
        }
    } break;
    }
    return 0;
}
