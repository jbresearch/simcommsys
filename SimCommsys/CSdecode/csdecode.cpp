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
#include "commsys_fulliter.h"
#include "commsys_stream.h"
#include "channel_stream.h"
#include "codec/codec_softout.h"
#include "cputimer.h"

#include <boost/program_options.hpp>
#include <iostream>
#include <list>

namespace csdecode {

// block reading

template <class S>
void read(std::istream& sin, libbase::vector<S>& result,
      const libbase::size_type<libbase::vector>& blocksize)
   {
   std::list<S> items;
   // Repeat until all elements read or end of stream
   for (int i = 0; i < blocksize && !sin.eof(); i++)
      {
      S x;
      sin >> libbase::eatcomments >> x;
      assertalways(sin.good() || sin.eof());
      items.push_back(x);
      }
   std::cerr << "Read block of length = " << items.size() << std::endl;
   // copy to required object type
   result.init(items.size());
   typename std::list<S>::iterator i;
   int j;
   for (i = items.begin(), j = 0; i != items.end(); i++, j++)
      result(j) = *i;
   }

template <class S>
void readsingleblock(std::istream& sin, libbase::vector<S>& result,
      const libbase::size_type<libbase::vector>& blocksize)
   {
   std::list<S> items;
   // Skip any preceding comments and whitespace
   sin >> libbase::eatcomments;
   // Repeat until end of stream
   while (!sin.eof())
      {
      S x;
      sin >> x >> libbase::eatcomments;
      assertalways(sin.good() || sin.eof());
      items.push_back(x);
      }
   std::cerr << "Read block of length = " << items.size() << std::endl;
   // truncate if necessary
   if (blocksize > 0 && blocksize < int(items.size()))
      {
      items.resize(blocksize);
      std::cerr << "Truncated to length = " << items.size() << std::endl;
      }
   // copy to required object type
   result.init(items.size());
   typename std::list<S>::iterator i;
   int j;
   for (i = items.begin(), j = 0; i != items.end(); i++, j++)
      result(j) = *i;
   }

template <class S>
void readsingleblock(std::istream& sin, libbase::matrix<S>& result,
      const libbase::size_type<libbase::matrix>& blocksize)
   {
   failwith("not implemented");
   }

template <class S, template <class > class C>
void readnextblock(std::istream& sin, C<S>& result,
      const libbase::size_type<C>& blocksize)
   {
   result.init(blocksize);
   result.serialize(sin);
   }

// block read and receive methods

template <class S, template <class > class C>
void receiver_soft_single(std::istream& sin, libcomm::commsys<S, C>* system,
      const libbase::size_type<C>& blocksize)
   {
   failwith("Not supported.");
   }

template <class S, template <class > class C>
void receiver_soft_multi(std::istream& sin, libcomm::commsys<S, C>* system,
      const libbase::size_type<C>& blocksize)
   {
   typedef libbase::vector<double> array1d_t;
   C<array1d_t> ptable_in;
   readnextblock(sin, ptable_in, blocksize);
   system->softreceive_path(ptable_in);
   }

template <class S, template <class > class C>
void receiver_single(std::istream& sin, libcomm::commsys<S, C>* system,
      const libbase::size_type<C>& blocksize)
   {
   C<S> received;
   readsingleblock(sin, received, blocksize);
   system->receive_path(received);
   }

template <class S, template <class > class C>
void receiver_multi(std::istream& sin, libcomm::commsys<S, C>* system,
      const libbase::size_type<C>& blocksize)
   {
   C<S> received;
   readnextblock(sin, received, blocksize);
   system->receive_path(received);
   }

//template <class S, template <class > class C>
//void receiver_multi_stream(std::istream& sin,
//      libcomm::commsys_stream<S, C>* system,
//      const libbase::size_type<C>& blocksize)

template <class S>
void receiver_multi_stream(std::istream& sin, libcomm::commsys_stream<S,
      libbase::vector, float>* system,
      const libbase::size_type<libbase::vector>& blocksize)
   {
   // Keep posterior probabilities at end-of-frame and computed drift
   static libbase::vector<double> eof_post;
   static libbase::size_type<libbase::vector> offset;
   static libbase::size_type<libbase::vector> estimated_drift;
   // Keep received sequence
   static libbase::vector<S> received;

   // Shorthand for transmitted frame size
   const int tau = system->output_block_size();

   // Keep a copy of the last frame's offset (in case state space changes)
   const libbase::size_type<libbase::vector> oldoffset = offset;

   // Determine the suggested look-ahead quantity
   const libbase::size_type<libbase::vector> lookahead =
         system->getmodem_stream().get_suggested_lookahead();
   // Determine start-of-frame and end-of-frame probabilities
   libbase::vector<double> sof_prior;
   libbase::vector<double> eof_prior;
   system->compute_priors(eof_post, lookahead, sof_prior, eof_prior, offset);

   // Extract required segment of existing stream
   system->stream_advance(received, oldoffset, estimated_drift, offset);
   // Determine required segment size
   const int length = tau + lookahead + eof_prior.size() - 1
         - received.size();
#ifndef NDEBUG
   std::cerr << "DEBUG: this segment = " << length << std::endl;
#endif
   // Read required segment from file
   libbase::vector<S> received_next;
   read(sin, received_next, libbase::size_type<libbase::vector>(length));
   // Assemble received sequence
   received = concatenate(received, received_next);
   // Stop here if the received sequence is too short
   if (received.size() < tau)
      {
      std::cerr << "Received sequence too short, stopping here." << std::endl;
      exit(1);
      }
   // Handle short received sequences
   else if (received.size() < tau + eof_prior.size() - 1)
      {
      libbase::vector<S> padding;
      padding.init(tau + eof_prior.size() - 1 - received.size());
      padding = 0; // value is irrelevant as this is not used
      received = concatenate(received, padding);
      const int start = eof_prior.size() - padding.size() - 1;
      const int n = padding.size();
      eof_prior.segment(start, n) = 0;
#ifndef NDEBUG
      std::cerr << "DEBUG: padding size = " << padding.size() << std::endl;
      std::cerr << "DEBUG: eof_prior = " << eof_prior << std::endl;
#endif
      }

   // Demodulate -> Inverse Map -> Translate
   system->receive_path(received, lookahead, sof_prior, eof_prior, offset);
   // Store posterior end-of-frame drift probabilities
   eof_post = system->get_eof_post();

   // Determine estimated drift
   estimated_drift = libcomm::commsys_stream<S, libbase::vector, float>::estimate_drift(eof_post,
         offset);
   // Centralize posterior probabilities
   eof_post = libcomm::commsys_stream<S, libbase::vector, float>::centralize_pdf(eof_post,
         estimated_drift);
   }

template <class S>
void receiver_multi_stream(std::istream& sin, libcomm::commsys_stream<S,
      libbase::matrix, float>* system,
      const libbase::size_type<libbase::matrix>& blocksize)
   {
   failwith("Not implemented.");
   }

// results output methods

template <class S, template <class > class C>
void decode_soft(std::ostream& sout, libcomm::commsys<S, C>* system)
   {
   typedef libbase::vector<double> array1d_t;
   typedef libcomm::codec_softout<C> codec_so;
   codec_so& cdc = dynamic_cast<codec_so&> (*system->getcodec());
   C<array1d_t> ptable_out;
   for (int i = 0; i < system->num_iter(); i++)
      cdc.softdecode(ptable_out);
   ptable_out.serialize(sout);
   }

template <class S, template <class > class C>
void decode(std::ostream& sout, libcomm::commsys<S, C>* system)
   {
   C<int> decoded;
   for (int i = 0; i < system->num_iter(); i++)
      system->decode(decoded);
   decoded.serialize(sout, '\n');
   }

/*!
 * \brief   Main process
 *
 * Reads the supplied system from file, and decodes from given input to
 * output stream.
 */

template <class S, template <class > class C>
void process(const std::string& fname, double p, bool softin, bool softout,
      bool knownend, int count, libbase::size_type<C>& blocksize,
      std::istream& sin = std::cin, std::ostream& sout = std::cout)
   {
   // define types
   typedef libcomm::commsys<S, C> commsys;
   typedef libcomm::commsys_stream<S, C, float> commsys_stream;

   // Communication system
   commsys *system = libcomm::loadfromfile<commsys>(fname);
   std::cerr << system->description() << std::endl;
   // Set channel parameter
   system->getrxchan()->set_parameter(p);
   // Initialize system
   libbase::randgen r;
   r.seed(0);
   system->seedfrom(r);
   // Determine block size to use if necessary
   if (!knownend && blocksize == 0)
      blocksize = system->output_block_size();
   // Check if this is a stream-oriented system
   commsys_stream* system_stream = dynamic_cast<commsys_stream*> (system);

   // Repeat until required number of blocks read or end of stream
   bool ready = false;
   for (int i = 0; !ready;)
      {
      // read next frame and pass through receiver
      if (softin)
         {
         if (knownend)
            receiver_soft_single(sin, system, blocksize);
         else
            receiver_soft_multi(sin, system, blocksize);
         }
      else
         {
         if (knownend)
            receiver_single(sin, system, blocksize);
         else
            {
            if (system_stream)
               receiver_multi_stream(sin, system_stream, blocksize);
            else
               receiver_multi(sin, system, blocksize);
            }
         }
      libbase::eatwhite(sin);
      // decode and output result
      if (softout)
         decode_soft(sout, system);
      else
         decode(sout, system);
      // loop advance
      i++;
      ready = (count > 0) ? (i >= count) : sin.eof();
      }

   // Destroy what was created on the heap
   delete system;
   }

/*!
 * \brief   Communication Systems Decoder
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
   desc.add_options()("soft-in,s", po::bool_switch(), "enable soft input");
   desc.add_options()("soft-out,o", po::bool_switch(), "enable soft output");
   desc.add_options()("known-end,k", po::bool_switch(),
         "enable known end-of-frame for last frame in stream-oriented systems");
   desc.add_options()("block-count,n", po::value<int>()->default_value(0),
         "number of blocks to read (default: read until end-of-file)");
   desc.add_options()("block-length", po::value<int>(),
         "block length to read for vector container (default: tx size)");
   desc.add_options()("row-size", po::value<int>(),
         "row size to read for matrix container (default: tx size)");
   desc.add_options()("col-size", po::value<int>(),
         "column size to read for matrix container (default: tx size)");
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
   const bool softin = vm["soft-in"].as<bool> ();
   const bool softout = vm["soft-out"].as<bool> ();
   const bool knownend = vm["known-end"].as<bool> ();
   const int count = vm["block-count"].as<int> ();
   // Check for compatibility
   if (knownend && count != 1)
      failwith("Known-end only implemented for single-block input.");

   // Main process
   if (container == "vector")
      {
      // determine block size to pass
      libbase::size_type<libbase::vector> blocksize;
      if (vm.count("block-length"))
         {
         const int length = vm["block-length"].as<int> ();
         blocksize = libbase::size_type<libbase::vector>(length);
         }
      // call main process with correct template parameters
      using libbase::vector;
      using libbase::gf;
      using libbase::erasable;
      using libcomm::sigspace;
      if (type == "erasable<bool>")
         process<erasable<bool>, vector> (filename, parameter, softin, softout, knownend,
               count, blocksize);
      else if (type == "bool")
         process<bool, vector> (filename, parameter, softin, softout, knownend,
               count, blocksize);
      else if (type == "gf2")
         process<gf<1, 0x3> , vector> (filename, parameter, softin, softout,
               knownend, count, blocksize);
      else if (type == "gf4")
         process<gf<2, 0x7> , vector> (filename, parameter, softin, softout,
               knownend, count, blocksize);
      else if (type == "gf8")
         process<gf<3, 0xB> , vector> (filename, parameter, softin, softout,
               knownend, count, blocksize);
      else if (type == "gf16")
         process<gf<4, 0x13> , vector> (filename, parameter, softin, softout,
               knownend, count, blocksize);
      else if (type == "gf32")
         process<gf<5, 0x25> , vector> (filename, parameter, softin, softout,
               knownend, count, blocksize);
      else if (type == "gf64")
         process<gf<6, 0x43> , vector> (filename, parameter, softin, softout,
               knownend, count, blocksize);
      else if (type == "gf128")
         process<gf<7, 0x89> , vector> (filename, parameter, softin, softout,
               knownend, count, blocksize);
      else if (type == "gf256")
         process<gf<8, 0x11D> , vector> (filename, parameter, softin, softout,
               knownend, count, blocksize);
      else if (type == "gf512")
         process<gf<9, 0x211> , vector> (filename, parameter, softin, softout,
               knownend, count, blocksize);
      else if (type == "gf1024")
         process<gf<10, 0x409> , vector> (filename, parameter, softin, softout,
               knownend, count, blocksize);
      else if (type == "sigspace")
         process<sigspace, vector> (filename, parameter, softin, softout,
               knownend, count, blocksize);
      else
         {
         std::cerr << "Unrecognized symbol type: " << type << std::endl;
         return 1;
         }
      }
   else if (container == "matrix")
      {
      // determine block size to pass
      libbase::size_type<libbase::matrix> blocksize;
      if (vm.count("row-size") && vm.count("col-size"))
         {
         const int rows = vm["row-size"].as<int> ();
         const int cols = vm["col-size"].as<int> ();
         blocksize = libbase::size_type<libbase::matrix>(rows, cols);
         }
      // call main process with correct template parameters
      using libbase::matrix;
      using libbase::gf;
      using libcomm::sigspace;
      if (type == "bool")
         process<bool, matrix> (filename, parameter, softin, softout, knownend,
               count, blocksize);
      else if (type == "gf2")
         process<gf<1, 0x3> , matrix> (filename, parameter, softin, softout,
               knownend, count, blocksize);
      else if (type == "gf4")
         process<gf<2, 0x7> , matrix> (filename, parameter, softin, softout,
               knownend, count, blocksize);
      else if (type == "gf8")
         process<gf<3, 0xB> , matrix> (filename, parameter, softin, softout,
               knownend, count, blocksize);
      else if (type == "gf16")
         process<gf<4, 0x13> , matrix> (filename, parameter, softin, softout,
               knownend, count, blocksize);
      else if (type == "gf32")
         process<gf<5, 0x25> , matrix> (filename, parameter, softin, softout,
               knownend, count, blocksize);
      else if (type == "gf64")
         process<gf<6, 0x43> , matrix> (filename, parameter, softin, softout,
               knownend, count, blocksize);
      else if (type == "gf128")
         process<gf<7, 0x89> , matrix> (filename, parameter, softin, softout,
               knownend, count, blocksize);
      else if (type == "gf256")
         process<gf<8, 0x11D> , matrix> (filename, parameter, softin, softout,
               knownend, count, blocksize);
      else if (type == "gf512")
         process<gf<9, 0x211> , matrix> (filename, parameter, softin, softout,
               knownend, count, blocksize);
      else if (type == "gf1024")
         process<gf<10, 0x409> , matrix> (filename, parameter, softin, softout,
               knownend, count, blocksize);
      else if (type == "sigspace")
         process<sigspace, matrix> (filename, parameter, softin, softout,
               knownend, count, blocksize);
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
   return csdecode::main(argc, argv);
   }
