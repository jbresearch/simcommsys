/*!
   \file
   \brief   Communication Systems Decoder
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "serializer_libcomm.h"
#include "commsys.h"
#include "timer.h"

#include <boost/program_options.hpp>
#include <iostream>

template <class S>
libcomm::commsys<S> *createsystem(const std::string& fname)
   {
   const libcomm::serializer_libcomm my_serializer_libcomm;
   // load system from string representation
   libcomm::commsys<S> *system;
   std::ifstream file(fname.c_str());
   file >> system;
   // check for errors in loading system
   libbase::verifycompleteload(file);
   return system;
   }

template <class S>
void process(const std::string& fname, double p, std::istream& sin, std::ostream& sout)
   {
   // Communication system
   libcomm::commsys<S> *system = createsystem<S>(fname);
   std::cerr << system->description() << "\n";
   // Set channel parameter
   system->getchan()->set_parameter(p);
   // Repeat until end of stream
   while(!sin.eof())
      {
      libbase::vector<S> received(system->output_block_size());
      received.serialize(sin);
      system->translate(received);
      libbase::vector<int> decoded;
      for(int i=0; i<system->getcodec()->num_iter(); i++)
         system->getcodec()->decode(decoded);
      decoded.serialize(sout, '\n');
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
      ;
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   po::notify(vm);

   // Validate user parameters
   if(vm.count("help"))
      {
      std::cout << desc << "\n";
      return 1;
      }

   // Main process
   process<bool>(vm["system-file"].as<std::string>(),
      vm["parameter"].as<double>(), std::cin, std::cout);

   return 0;
   }
