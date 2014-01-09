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

namespace testlinearity {

namespace po = boost::program_options;

template <template <class > class C>
C<int> createsource(libbase::random& r, const libcomm::codec<C, double> *cdc)
   {
   const int tau = cdc->input_block_size();
   C<int> source(tau);
   for (int t = 0; t < tau; t++)
      source(t) = r.ival(cdc->num_inputs());
   return source;
   }

template <class S, template <class > class C>
void process(const std::string& fname, int count)
   {
   // Communication system
   libcomm::codec<C, double> *cdc = libcomm::loadfromfile<libcomm::codec<C,
         double> >(fname);
   std::cerr << cdc->description() << std::endl;
   // Initialize system
   libbase::randgen r;
   r.seed(0);
   cdc->seedfrom(r);

   // Repeat a few times
   for (int i = 0; i < count; i++)
      {
      // Create two source sequences & their sum
      C<int> source1 = createsource(r, cdc);
      C<int> source2 = createsource(r, cdc);
      C<int> source3 = C<int> (C<S> (source1) + C<S> (source2));
      // Determine corresponding encoded sequences
      C<int> encoded1;
      cdc->encode(source1, encoded1);
      C<int> encoded2;
      cdc->encode(source2, encoded2);
      C<int> encoded3;
      cdc->encode(source3, encoded3);
      // Verify that the sum of codewords is still a codeword
      C<int> encodedS = C<int> (C<S> (encoded1) + C<S> (encoded2));
      assertalways(encodedS.isequalto(encoded3));
      }
   }

/*!
 * \brief   Test linearity properties of codec in given system
 * \author  Johann Briffa
 */

int main(int argc, char *argv[])
   {
   libbase::cputimer tmain("Main timer");

   // Set up user parameters
   po::options_description desc("Allowed options");
   desc.add_options()("help", "print this help message");
   desc.add_options()("system-file,i", po::value<std::string>(),
         "input file containing codec description");
   desc.add_options()("type,t",
         po::value<std::string>()->default_value("bool"), "codec symbol type");
   desc.add_options()("container,c", po::value<std::string>()->default_value(
         "vector"), "input/output container type");
   desc.add_options()("length,n", po::value<int>()->default_value(100),
         "test length (number of codeword pairs to test)");
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   po::notify(vm);

   // Validate user parameters
   if (vm.count("help") || vm.count("system-file") == 0)
      {
      std::cerr << desc << std::endl;
      return 1;
      }
   // Shorthand access for parameters
   const std::string container = vm["container"].as<std::string> ();
   const std::string type = vm["type"].as<std::string> ();
   const std::string filename = vm["system-file"].as<std::string> ();
   const int n = vm["length"].as<int> ();

   // Main process
   if (container == "vector")
      {
      using libbase::vector;
      using libbase::gf;
      using libcomm::sigspace;
      if (type == "bool")
         process<bool, vector> (filename, n);
      else if (type == "gf2")
         process<gf<1, 0x3> , vector> (filename, n);
      else if (type == "gf4")
         process<gf<2, 0x7> , vector> (filename, n);
      else if (type == "gf8")
         process<gf<3, 0xB> , vector> (filename, n);
      else if (type == "gf16")
         process<gf<4, 0x13> , vector> (filename, n);
      else if (type == "gf32")
         process<gf<5, 0x25> , vector> (filename, n);
      else if (type == "gf64")
         process<gf<6, 0x43> , vector> (filename, n);
      else if (type == "gf128")
         process<gf<7, 0x89> , vector> (filename, n);
      else if (type == "gf256")
         process<gf<8, 0x11D> , vector> (filename, n);
      else if (type == "gf512")
         process<gf<9, 0x211> , vector> (filename, n);
      else if (type == "gf1024")
         process<gf<10, 0x409> , vector> (filename, n);
      else if (type == "int")
         process<int, vector> (filename, n);
      else
         {
         std::cerr << "Unrecognized symbol type: " << type << std::endl;
         return 1;
         }
      }
   //   else if (container == "matrix")
   //      {
   //      using libbase::matrix;
   //      using libbase::gf;
   //      using libcomm::sigspace;
   //      if (type == "bool")
   //         process<bool, matrix> (filename,n);
   //      else if (type == "gf2")
   //         process<gf<1, 0x3> , matrix> (filename,n);
   //      else if (type == "gf4")
   //         process<gf<2, 0x7> , matrix> (filename,n);
   //      else if (type == "gf8")
   //         process<gf<3, 0xB> , matrix> (filename,n);
   //      else if (type == "gf16")
   //         process<gf<4, 0x13> , matrix> (filename,n);
   //      else if (type == "gf32")
   //         process<gf<5, 0x25> , matrix> (filename,n);
   //      else if (type == "gf64")
   //         process<gf<6, 0x43> , matrix> (filename,n);
   //      else if (type == "gf128")
   //         process<gf<7, 0x89> , matrix> (filename,n);
   //      else if (type == "gf256")
   //         process<gf<8, 0x11D> , matrix> (filename,n);
   //      else if (type == "gf512")
   //         process<gf<9, 0x211> , matrix> (filename,n);
   //      else if (type == "gf1024")
   //         process<gf<10, 0x409> , matrix> (filename,n);
   //      else if (type == "sigspace")
   //         process<sigspace, matrix> (filename,n);
   //      else
   //         {
   //         std::cerr << "Unrecognized symbol type: " << type << std::endl;
   //         return 1;
   //         }
   //      }
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
   return testlinearity::main(argc, argv);
   }
