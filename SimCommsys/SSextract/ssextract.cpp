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
#include "hard_decision.h"
#include "cputimer.h"

#include <boost/program_options.hpp>
#include <iostream>
#include <fstream>
#include <typeinfo>

namespace ssextract {

template <class S>
libimage::image<S> loadimage(std::istream& sin)
   {
   // load image from stream
   libimage::image<S> im;
   im.serialize(sin);
   libbase::verifycomplete(sin);
   return im;
   }

/*!
 * \brief   Main data-extraction process
 * \author  Johann Briffa
 *
 * This method extracts an embedded data sequence from a given stego-medium;
 * the stego-system used is described by a system file. The stego-medium is
 * supplied through an input stream. The data sequence is output to a stream.
 * The use of streams facilitates chaining this tool with others along the
 * transmission path.
 *
 * \note The stego-medium size and shape must be exactly that required to
 * fit the embedder.
 *
 * \note Currently only images are supported as host media; extension to other
 * media requires the creation of a stego-system object.
 */

template <class S, template <class > class C>
void process(const std::string& systemfile, const std::string& channelfile,
      double p, bool softout, std::istream& sin = std::cin, std::ostream& sout =
            std::cout)
   {
   // define types
   typedef libbase::vector<double> array1d_t;

   // Load stego-medium
   libimage::image<S> stegoimage = loadimage<S> (sin);
   // Stego-system embedder
   libcomm::blockembedder<S, C> *system = libcomm::loadfromfile<
         libcomm::blockembedder<S, C> >(systemfile);
   std::cerr << system->description() << std::endl;
   // Channel model
   libcomm::channel<S, C> *chan =
         libcomm::loadfromfile<libcomm::channel<S, C> >(channelfile);
   std::cerr << chan->description() << std::endl;
   // Set channel parameter
   chan->set_parameter(p);
   // Initialize system
   libbase::randgen r;
   r.seed(0);
   system->seedfrom(r);
   system->set_blocksize(stegoimage.size());
   // Set up and initialize hard-decision box
   libcomm::hard_decision<C, double, int> hd_functor;
   hd_functor.seedfrom(r);
   // Repeat for all image channels
   for (int c = 0; c < stegoimage.channels(); c++)
      {
      // Extract channel
      C<S> stego = stegoimage.getchannel(c);
      // Extract message
      C<array1d_t> ptable;
      system->extract(*chan, stego, ptable);
      // Output results
      if (softout)
         ptable.serialize(sout);
      else
         {
         C<int> decoded;
         hd_functor(ptable, decoded);
         decoded.serialize(sout, '\n');
         }
      }
   // Verify that there is no pending data
   sin >> libbase::verifycomplete;
   // Destroy what was created on the heap
   delete system;
   delete chan;
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
   // TODO: integrate channel within system file describing stego-system
   desc.add_options()("channel-file,h", po::value<std::string>(),
         "input file containing channel description");
   desc.add_options()("type,t", po::value<std::string>()->default_value("int"),
         "host symbol type");
   desc.add_options()("container,c", po::value<std::string>()->default_value(
         "matrix"), "input/output container type");
   desc.add_options()("parameter,r", po::value<double>(), "channel parameter");
   //desc.add_options()("soft-in,s", po::bool_switch(), "enable soft input");
   desc.add_options()("soft-out,o", po::bool_switch(), "enable soft output");
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   po::notify(vm);

   // Validate user parameters
   if (vm.count("help") || vm.count("system-file") == 0 || vm.count(
         "channel-file") == 0 || vm.count("parameter") == 0)
      {
      std::cerr << desc << std::endl;
      return 1;
      }
   // Shorthand access for parameters
   const std::string container = vm["container"].as<std::string> ();
   const std::string type = vm["type"].as<std::string> ();
   const std::string systemfile = vm["system-file"].as<std::string> ();
   const std::string channelfile = vm["channel-file"].as<std::string> ();
   const double parameter = vm["parameter"].as<double> ();
   //const bool softin = vm["soft-in"].as<bool> ();
   const bool softout = vm["soft-out"].as<bool> ();

   // Main process
   /* TODO: add support for vector
    if (container == "vector")
    {
    using libbase::vector;
    if (type == "int")
    process<int, vector> (systemfile, channelfile, parameter, softout);
    else if (type == "float")
    process<float, vector> (systemfile, channelfile, parameter, softout);
    else if (type == "double")
    process<double, vector> (systemfile, channelfile, parameter, softout);
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
         process<int, matrix> (systemfile, channelfile, parameter, softout);
      else if (type == "float")
         process<float, matrix> (systemfile, channelfile, parameter, softout);
      else if (type == "double")
         process<double, matrix> (systemfile, channelfile, parameter, softout);
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
   return ssextract::main(argc, argv);
   }
