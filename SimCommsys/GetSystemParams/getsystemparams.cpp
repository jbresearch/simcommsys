/*!
 * \file
 *
 * Copyright (c) 2025 Mark Mizzi
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

#include "commsys.h"
#include "config.h"
#include "gf.h"
#include "matrix.h"
#include "serializer_libcomm.h"
#include "sigspace.h"
#include "vector.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/program_options.hpp>
#include <nlohmann/json.hpp>

template <class S, template <class> class C>
void
process(const std::string& fname,
        std::vector<std::string>& required_params,
        std::ostream& sout)
{
    // Communication system
    std::shared_ptr<libcomm::commsys<S, C>> system =
        libcomm::loadfromfile<libcomm::commsys<S, C>>(fname);

    nlohmann::json params;
    params["system"] = fname;

    for (const std::string& param : required_params) {
        if (param == "input-alphabetsize") {
            params["input-alphabetsize"] = system->num_inputs();
        } else if (param == "output-alphabetsize") {
            params["output-alphabetsize"] = system->num_outputs();
        } else {
            std::cerr << "WARN: Got unrecognized parameter " << param
                      << std::endl;
        }
    }

    sout << params.dump(1, '\t', true) << std::flush;
}

/*!
 * \brief   Get information about a system file.
 * \author  Mark Mizzi
 */
int
main(int argc, char* argv[])
{
    // Set up user parameters
    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()("help", "print this help message");
    desc.add_options()("system-file,i",
                       po::value<std::string>()->required(),
                       "input file containing system description");
    desc.add_options()("param,p",
                       po::value<std::vector<std::string>>()->multitoken(),
                       "system parameter(s) to extract from system file");
    desc.add_options()("output-file,o",
                       po::value<std::string>(),
                       "output file to hold system parameters. If not given "
                       "parameters are printed to stdout.");
    desc.add_options()("type,t",
                       po::value<std::string>()->default_value("bool"),
                       "modulation symbol type");
    desc.add_options()("container,c",
                       po::value<std::string>()->default_value("vector"),
                       "input/output container type");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // Validate user parameters
    if (vm.count("help") || vm.count("system-file") == 0) {
        std::cerr << desc << std::endl;
        return 1;
    }

    std::string sysfilename = vm["system-file"].as<std::string>();

    std::ofstream outfile;
    std::ostream* out =
        &std::cout; // store reference to output stream regardless of whether
                    // this is stdout or a file.
    if (vm.count("output-file")) {
        std::string outfilename = vm["output-file"].as<std::string>();
        outfile.open(outfilename, std::ios::out);
        if (!outfile.is_open()) {
            std::cerr << "Could not open given output file " << outfilename
                      << std::endl;
            exit(-1);
        }

        out = &outfile;
    }

    std::vector<std::string> required_params =
        vm["param"].as<std::vector<std::string>>();

    // Shorthand access for parameters
    const std::string container = vm["container"].as<std::string>();
    const std::string type = vm["type"].as<std::string>();

    // we need to remove namespace qualifiers from containers and types as we
    // will be using them in macros (and stringifying them and comparing them to
    // user args).
    using libbase::matrix;
    using libbase::vector;
    using libcomm::sigspace;
#define USING_GF(r, x, type) using libbase::type;
    BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

    // Explicit if conditions so that we can dynamically map container and
    // symbol type given as user args to a template function.
#define CONTAINER_TYPE_SEQ (vector)(matrix)
#define SYMBOL_TYPE_SEQ (bool)GF_TYPE_SEQ(sigspace)

#define PROCESS(r, args)                                                       \
    if (container == BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) &&                                           \
                         type ==                                               \
                             BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args))) {  \
        process<BOOST_PP_SEQ_ELEM(1, args), BOOST_PP_SEQ_ELEM(0, args)>(       \
            sysfilename, required_params, *out);                               \
        return 0;                                                              \
    }

    BOOST_PP_SEQ_FOR_EACH_PRODUCT(PROCESS,
                                  (CONTAINER_TYPE_SEQ)(SYMBOL_TYPE_SEQ))

    std::cerr << "Unrecognized container or symbol type: " << container << ", "
              << type << std::endl;
    return 1;
}