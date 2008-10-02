/*!
   \file
   \brief   Communication Systems Encoder
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
   // load system from string representation
   libcomm::commsys<S> *system;
   std::ifstream file(fname.c_str());
   file >> system;
   // check for errors in loading system
   libbase::verifycompleteload(file);
   return system;
   }

template <class S>
void process(const std::string& fname, std::istream& sin, std::ostream& sout)
   {
   // Communication system
   libcomm::commsys<S> *system = createsystem<S>(fname);
   std::cerr << system->description() << "\n";
   // Repeat until end of stream
   while(!sin.eof())
      {
      libbase::vector<int> source(system->input_block_size());
      source.serialize(sin);
      libbase::vector<S> transmitted = system->encode(source);
      transmitted.serialize(sout, '\n');
      libbase::eatwhite(sin);
      }
   }

int main(int argc, char *argv[])
   {
   const libcomm::serializer_libcomm my_serializer_libcomm;

   libbase::timer tmain("Main timer");

   // Set up user parameters
   namespace po = boost::program_options;
   po::options_description desc("Allowed options");
   desc.add_options()
      ("help", "print this help message")
      ("system-file,i", po::value<std::string>(),
         "input file containing system description")
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
   process<bool>(vm["system-file"].as<std::string>(), std::cin, std::cout);

   return 0;
   }
