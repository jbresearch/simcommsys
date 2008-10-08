/*!
   \file
   \brief   Communication Systems Encoder-Transmit-Decoder Cycle
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "serializer_libcomm.h"
#include "commsys.h"
#include "codec_softout.h"
#include "timer.h"

#include <boost/program_options.hpp>
#include <iostream>

template <class S>
void process(const std::string& fname, double p, bool soft, std::istream& sin, std::ostream& sout)
   {
   // Communication system
   libcomm::commsys<S> *system = libcomm::loadfromfile< libcomm::commsys<S> >(fname);
   std::cerr << system->description() << "\n";
   // Set channel parameter
   system->getchan()->set_parameter(p);
   // Initialize system
   libbase::randgen r;
   r.seed(0);
   system->seedfrom(r);
   // Repeat until end of stream
   while(!sin.eof())
      {
      libbase::vector<int> source(system->input_block_size());
      source.serialize(sin);
      libbase::vector<S> transmitted = system->encode(source);
      libbase::vector<S> received;
      system->getchan()->transmit(transmitted, received);
      system->translate(received);
      if(soft)
         {
         libcomm::codec_softout<double>& cdc = 
            dynamic_cast< libcomm::codec_softout<double>& >(*system->getcodec());
         libbase::matrix<double> ptable;
         for(int i=0; i<system->getcodec()->num_iter(); i++)
            cdc.decode(ptable);
         ptable.transpose().serialize(sout);
         }
      else
         {
         libbase::vector<int> decoded;
         for(int i=0; i<system->getcodec()->num_iter(); i++)
            system->getcodec()->decode(decoded);
         decoded.serialize(sout, '\n');
         }
      libbase::eatwhite(sin);
      }
   }

int main(int argc, char *argv[])
   {
   libbase::timer tmain("Main timer");

   // Set up user parameters
   namespace po = boost::program_options;
   po::options_description desc("Allowed options");
   desc.add_options()
      ("help", "print this help message")
      ("system-file,i", po::value<std::string>(),
         "input file containing system description")
      ("parameter,p", po::value<double>(),
         "channel parameter")
      ("soft-out,s", po::bool_switch(),
         "enable soft output")
      ;
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   po::notify(vm);

   // Validate user parameters
   if(vm.count("help"))
      {
      std::cerr << desc << "\n";
      return 1;
      }

   // Main process
   process<bool>(vm["system-file"].as<std::string>(),
      vm["parameter"].as<double>(), vm["soft-out"].as<bool>(),
      std::cin, std::cout);

   return 0;
   }
