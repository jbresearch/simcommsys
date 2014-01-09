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

#include "serializer_libcomm.h"
#include "cputimer.h"

#include <boost/program_options.hpp>
#include <iostream>

namespace inspectserializer {

/*!
 * \brief   Serializer Class Support Inspector
 * \author  Johann Briffa
 *
 * This program returns a list of base classes supported by the serializer.
 * If a base class is given, then this program returns supported derived
 * classes.
 */

int main(int argc, char *argv[])
   {
   libbase::cputimer tmain("Main timer");

   // Set up user parameters
   namespace po = boost::program_options;
   po::options_description desc("Allowed options");
   desc.add_options()("help,h", "print this help message");
   desc.add_options()("base,b", po::value<std::string>(), "base class");
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   po::notify(vm);

   // Validate user parameters
   if (vm.count("help"))
      {
      std::cerr << desc << std::endl;
      return 1;
      }

   // Main process

   // Make sure we instantiate everything
   const libcomm::serializer_libcomm my_serializer_libcomm;
   // Get required list
   std::list<std::string> result;
   if (vm.count("base"))
      {
      // Shorthand access for parameters
      const std::string base = vm["base"].as<std::string> ();
      // Get list of derived classes
      result = libbase::serializer::get_derived_classes(base);
      // Print header
      std::cout << "List of derived classes:" << std::endl;
      }
   else
      {
      // Get list of base classes
      result = libbase::serializer::get_base_classes();
      // Print header
      std::cout << "List of base classes:" << std::endl;
      }
   // Print the list of classes
   for (std::list<std::string>::const_iterator i = result.begin(); i
         != result.end(); i++)
      std::cout << "\t" << *i << std::endl;

   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return inspectserializer::main(argc, argv);
   }
