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

namespace cstransmit {

template <class S, template <class > class C>
void process(const std::string& fname, double p, std::istream& sin = std::cin,
      std::ostream& sout = std::cout)
   {
   // Communication system
   libcomm::commsys<S, C> *system = libcomm::loadfromfile<
         libcomm::commsys<S, C> >(fname);
   std::cerr << system->description() << std::endl;
   // Set channel parameter
   system->gettxchan()->set_parameter(p);
   // Initialize system
   libbase::randgen r;
   r.seed(0);
   system->seedfrom(r);
   // Repeat until end of stream
   while (!sin.eof())
      {
      C<S> transmitted(system->output_block_size());
      transmitted.serialize(sin);
      C<S> received = system->transmit(transmitted);
      received.serialize(sout, '\n');
      libbase::eatwhite(sin);
      }
   // Destroy what was created on the heap
   delete system;
   }

/*!
 * \brief   Communication Systems Transmitter
 * \author  Johann Briffa
 */

int main(int argc, char *argv[])
   {
   libbase::cputimer tmain("Main timer");

   // Set up user parameters
   namespace po = boost::program_options;
   po::options_description desc("Allowed options");
   desc.add_options()("help", "print this help message");
   desc.add_options()("system-file,i", po::value<std::string>(),
         "input file containing system description");
   desc.add_options()("type,t",
         po::value<std::string>()->default_value("bool"),
         "modulation symbol type");
   desc.add_options()("container,c", po::value<std::string>()->default_value(
         "vector"), "input/output container type");
   desc.add_options()("parameter,r", po::value<double>(), "channel parameter");
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   po::notify(vm);

   // Validate user parameters
   if (vm.count("help") || vm.count("system-file") == 0
         || vm.count("parameter") == 0)
      {
      std::cerr << desc << std::endl;
      return 1;
      }
   // Shorthand access for parameters
   const std::string container = vm["container"].as<std::string> ();
   const std::string type = vm["type"].as<std::string> ();
   const std::string filename = vm["system-file"].as<std::string> ();
   const double parameter = vm["parameter"].as<double> ();

   // Main process
   if (container == "vector")
      {
      using libbase::vector;
      using libbase::gf;
      using libcomm::sigspace;
      if (type == "bool")
         process<bool, vector> (filename, parameter);
      else if (type == "gf2")
         process<gf<1, 0x3> , vector> (filename, parameter);
      else if (type == "gf4")
         process<gf<2, 0x7> , vector> (filename, parameter);
      else if (type == "gf8")
         process<gf<3, 0xB> , vector> (filename, parameter);
      else if (type == "gf16")
         process<gf<4, 0x13> , vector> (filename, parameter);
      else if (type == "gf32")
         process<gf<5, 0x25> , vector> (filename, parameter);
      else if (type == "gf64")
         process<gf<6, 0x43> , vector> (filename, parameter);
      else if (type == "gf128")
         process<gf<7, 0x89> , vector> (filename, parameter);
      else if (type == "gf256")
         process<gf<8, 0x11D> , vector> (filename, parameter);
      else if (type == "gf512")
         process<gf<9, 0x211> , vector> (filename, parameter);
      else if (type == "gf1024")
         process<gf<10, 0x409> , vector> (filename, parameter);
      else if (type == "sigspace")
         process<sigspace, vector> (filename, parameter);
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
         process<bool, matrix> (filename, parameter);
      else if (type == "gf2")
         process<gf<1, 0x3> , matrix> (filename, parameter);
      else if (type == "gf4")
         process<gf<2, 0x7> , matrix> (filename, parameter);
      else if (type == "gf8")
         process<gf<3, 0xB> , matrix> (filename, parameter);
      else if (type == "gf16")
         process<gf<4, 0x13> , matrix> (filename, parameter);
      else if (type == "gf32")
         process<gf<5, 0x25> , matrix> (filename, parameter);
      else if (type == "gf64")
         process<gf<6, 0x43> , matrix> (filename, parameter);
      else if (type == "gf128")
         process<gf<7, 0x89> , matrix> (filename, parameter);
      else if (type == "gf256")
         process<gf<8, 0x11D> , matrix> (filename, parameter);
      else if (type == "gf512")
         process<gf<9, 0x211> , matrix> (filename, parameter);
      else if (type == "gf1024")
         process<gf<10, 0x409> , matrix> (filename, parameter);
      else if (type == "sigspace")
         process<sigspace, matrix> (filename, parameter);
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
   return cstransmit::main(argc, argv);
   }
