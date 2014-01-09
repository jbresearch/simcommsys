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
#include "image.h"
#include "blockembedder.h"
#include "filter/limitfilter.h"
#include "cputimer.h"

#include <boost/program_options.hpp>
#include <iostream>
#include <fstream>
#include <typeinfo>

namespace ssembed {

template <class S>
libimage::image<S> loadimage(const std::string& fname)
   {
   // load image from file
   std::ifstream file(fname.c_str(), std::ios_base::in | std::ios_base::binary);
   assertalways(file.is_open());
   libimage::image<S> im;
   im.serialize(file);
   libbase::verifycomplete(file);
   return im;
   }

/*!
 * \brief   Main data-embedding process
 * \author  Johann Briffa
 *
 * This method embeds a given data sequence into a given cover (host); the
 * stego-system used is described by a system file. The data sequence is
 * supplied through an input stream while the host medium is given as a file.
 * The stego-medium is output to a stream. The use of streams facilitates
 * chaining this tool with others along the transmission path.
 *
 * \note The input sequence size and shape must be exactly that required to
 * fit the given cover medium and embedder.
 *
 * \note Currently only images are supported as host media; extension to other
 * media requires the creation of a stego-system object.
 */

template <class S, template <class > class C>
void process(const std::string& systemfile, const std::string& hostfile,
      std::istream& sin = std::cin, std::ostream& sout = std::cout)
   {
   // Load host medium
   libimage::image<S> hostimage = loadimage<S> (hostfile);
   // Stego-system embedder
   libcomm::blockembedder<S, C> *system = libcomm::loadfromfile<
         libcomm::blockembedder<S, C> >(systemfile);
   std::cerr << system->description() << std::endl;
   // Initialize system
   libbase::randgen r;
   r.seed(0);
   system->seedfrom(r);
   system->set_blocksize(hostimage.size());
   // Create stego-medium
   libimage::image<S> stegoimage = hostimage;
   // Repeat for all image channels
   for (int c = 0; c < hostimage.channels(); c++)
      {
      // Extract channel
      C<S> host = hostimage.getchannel(c);
      // Read data sequence
      C<int> data(system->input_block_size());
      data.serialize(sin);
      // Embed
      C<S> stego;
      system->embed(system->num_symbols(), data, host, stego);
      // Limit values to usable range
      libimage::limitfilter<S> filter(hostimage.lo(), hostimage.hi());
      filter.apply(stego, stego);
      // Copy result into stego-image
      stegoimage.setchannel(c, stego);
      }
   // Save the resulting image
   stegoimage.serialize(sout);
   // Verify that there is no pending data
   sin >> libbase::verifycomplete;
   // Destroy what was created on the heap
   delete system;
   }

/*!
 * \brief   Stego-System Embedder
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
   desc.add_options()("host-file,h", po::value<std::string>(),
         "host medium file");
   desc.add_options()("type,t", po::value<std::string>()->default_value("int"),
         "host symbol type");
   desc.add_options()("container,c", po::value<std::string>()->default_value(
         "matrix"), "input/output container type");
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   po::notify(vm);

   // Validate user parameters
   if (vm.count("help") || vm.count("system-file") == 0
         || vm.count("host-file") == 0)
      {
      std::cerr << desc << std::endl;
      return 1;
      }
   // Shorthand access for parameters
   const std::string container = vm["container"].as<std::string> ();
   const std::string type = vm["type"].as<std::string> ();
   const std::string systemfile = vm["system-file"].as<std::string> ();
   const std::string hostfile = vm["host-file"].as<std::string> ();

   // Main process
   /* TODO: add support for vector
    if (container == "vector")
    {
    using libbase::vector;
    if (type == "int")
    process<int, vector> (systemfile, hostfile);
    else if (type == "float")
    process<float, vector> (systemfile, hostfile);
    else if (type == "double")
    process<double, vector> (systemfile, hostfile);
    else
    {
    std::cerr << "Unrecognized symbol type: " << type << std::endl;
    return 1;
    }
    }
    else */
   if (container == "matrix")
      {
      using libbase::matrix;
      if (type == "int")
         process<int, matrix> (systemfile, hostfile);
      else if (type == "float")
         process<float, matrix> (systemfile, hostfile);
      else if (type == "double")
         process<double, matrix> (systemfile, hostfile);
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
   return ssembed::main(argc, argv);
   }
