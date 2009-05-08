#include "serializer_libcomm.h"
#include "commsys.h"
#include "timer.h"

#include <boost/program_options.hpp>
#include <iostream>

template <class S>
void process(const std::string& fname, std::istream& sin, std::ostream& sout)
   {
   // Communication system
   libcomm::commsys<S> *system = libcomm::loadfromfile< libcomm::commsys<S> >(fname);
   std::cerr << system->description() << "\n";
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
      transmitted.serialize(sout, '\n');
      libbase::eatwhite(sin);
      }
   }

/*!
   \brief   Communication Systems Encoder
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   \bug Works only with bool systems!
*/

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
      ("type,t", po::value<std::string>()->default_value("bool"),
         "modulation symbol type")
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
   if(vm["type"].as<std::string>() == "bool")
      process<bool>(vm["system-file"].as<std::string>(), std::cin, std::cout);
   else if(vm["type"].as<std::string>() == "gf2")
      process< libbase::gf<1,0x3> >(vm["system-file"].as<std::string>(), std::cin, std::cout);
   else if(vm["type"].as<std::string>() == "gf4")
      process< libbase::gf<2,0x7> >(vm["system-file"].as<std::string>(), std::cin, std::cout);
   else if(vm["type"].as<std::string>() == "gf8")
      process< libbase::gf<3,0xB> >(vm["system-file"].as<std::string>(), std::cin, std::cout);
   else if(vm["type"].as<std::string>() == "gf16")
      process< libbase::gf<4,0x13> >(vm["system-file"].as<std::string>(), std::cin, std::cout);
   else if(vm["type"].as<std::string>() == "sigspace")
      process<libcomm::sigspace>(vm["system-file"].as<std::string>(), std::cin, std::cout);
   else
      {
      std::cerr << "Unrecognized symbol type: " << vm["type"].as<std::string>() << "\n";
      return 1;
      }

   return 0;
   }
