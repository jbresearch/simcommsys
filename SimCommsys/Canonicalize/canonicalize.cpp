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
#include "commsys.h"
#include "cputimer.h"

#include <boost/program_options.hpp>
#include <iostream>

namespace canonicalize {

template <class S, template <class > class C>
void process(std::istream& sin = std::cin, std::ostream& sout = std::cout)
   {
   // Read from input stream
   libcomm::commsys<S, C> *system = libcomm::loadandverify<libcomm::commsys<S,
         C> >(sin);
   // Write details on error stream
   std::cerr << system->description() << std::endl;
   // Write system in canonical form on output stream
   std::cout << system;
   // Destroy what was created on the heap
   delete system;
   }

/*!
 * \brief   Communication Systems File Canonicalizer
 * \author  Johann Briffa
 *
 * This program reads the system specified on standard input and rewrites
 * it to standard output in its canonical form.
 */

int main(int argc, char *argv[])
   {
   libbase::cputimer tmain("Main timer");

   // Set up user parameters
   namespace po = boost::program_options;
   po::options_description desc("Allowed options");
   desc.add_options()("help", "print this help message");
   desc.add_options()("type,t",
         po::value<std::string>()->default_value("bool"),
         "modulation symbol type");
   desc.add_options()("container,c", po::value<std::string>()->default_value(
         "vector"), "input/output container type");
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   po::notify(vm);

   // Validate user parameters
   if (vm.count("help"))
      {
      std::cerr << desc << std::endl;
      return 1;
      }
   // Shorthand access for parameters
   const std::string container = vm["container"].as<std::string> ();
   const std::string type = vm["type"].as<std::string> ();

   // Main process
   if (container == "vector")
      {
      using libbase::vector;
      using libbase::gf;
      using libbase::erasable;
      using libcomm::sigspace;
      if (type == "erasable<bool>")
         process<erasable<bool>, vector> ();
      else if (type == "bool")
         process<bool, vector> ();
      else if (type == "gf2")
         process<gf<1, 0x3> , vector> ();
      else if (type == "gf4")
         process<gf<2, 0x7> , vector> ();
      else if (type == "gf8")
         process<gf<3, 0xB> , vector> ();
      else if (type == "gf16")
         process<gf<4, 0x13> , vector> ();
      else if (type == "gf32")
         process<gf<5, 0x25> , vector> ();
      else if (type == "gf64")
         process<gf<6, 0x43> , vector> ();
      else if (type == "gf128")
         process<gf<7, 0x89> , vector> ();
      else if (type == "gf256")
         process<gf<8, 0x11D> , vector> ();
      else if (type == "gf512")
         process<gf<9, 0x211> , vector> ();
      else if (type == "gf1024")
         process<gf<10, 0x409> , vector> ();
      else if (type == "sigspace")
         process<sigspace, vector> ();
      else
         {
         std::cerr << "Unrecognized symbol type: " << type << std::endl;
         return 1;
         }
      }
   else if (container == "matrix")
      {
      using libbase::matrix;
      using libbase::gf;
      using libcomm::sigspace;
      if (type == "bool")
         process<bool, matrix> ();
      else if (type == "gf2")
         process<gf<1, 0x3> , matrix> ();
      else if (type == "gf4")
         process<gf<2, 0x7> , matrix> ();
      else if (type == "gf8")
         process<gf<3, 0xB> , matrix> ();
      else if (type == "gf16")
         process<gf<4, 0x13> , matrix> ();
      else if (type == "gf32")
         process<gf<5, 0x25> , matrix> ();
      else if (type == "gf64")
         process<gf<6, 0x43> , matrix> ();
      else if (type == "gf128")
         process<gf<7, 0x89> , matrix> ();
      else if (type == "gf256")
         process<gf<8, 0x11D> , matrix> ();
      else if (type == "gf512")
         process<gf<9, 0x211> , matrix> ();
      else if (type == "gf1024")
         process<gf<10, 0x409> , matrix> ();
      else if (type == "sigspace")
         process<sigspace, matrix> ();
      else
         {
         std::cerr << "Unrecognized symbol type: " << type << std::endl;
         return 1;
         }
      }
   else
      {
      std::cerr << "Unrecognized container type: " << container << std::endl;
      return 1;
      }

   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return canonicalize::main(argc, argv);
   }
