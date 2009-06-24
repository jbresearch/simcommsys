#include "serializer_libcomm.h"
#include "commsys.h"
#include "timer.h"

#include <boost/program_options.hpp>
#include <iostream>

namespace csencode {

template <class S, template<class> class C>
void process(const std::string& fname, std::istream& sin, std::ostream& sout)
   {
   // Communication system
   libcomm::commsys<S,C> *system = libcomm::loadfromfile< libcomm::commsys<S,C> >(fname);
   std::cerr << system->description() << "\n";
   // Initialize system
   libbase::randgen r;
   r.seed(0);
   system->seedfrom(r);
   // Repeat until end of stream
   while(!sin.eof())
      {
      C<int> source(system->input_block_size());
      source.serialize(sin);
      C<S> transmitted = system->encode(source);
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
      ("container,c", po::value<std::string>()->default_value("vector"),
         "input/output container type")
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
   if(vm["container"].as<std::string>() == "vector")
      {
      using libbase::vector;
      using libbase::gf;
      using libcomm::sigspace;
      if(vm["type"].as<std::string>() == "bool")
         process<bool,vector>(vm["system-file"].as<std::string>(), std::cin, std::cout);
      else if(vm["type"].as<std::string>() == "gf2")
         process< gf<1,0x3>,vector >(vm["system-file"].as<std::string>(), std::cin, std::cout);
      else if(vm["type"].as<std::string>() == "gf4")
         process< gf<2,0x7>,vector >(vm["system-file"].as<std::string>(), std::cin, std::cout);
      else if(vm["type"].as<std::string>() == "gf8")
         process< gf<3,0xB>,vector >(vm["system-file"].as<std::string>(), std::cin, std::cout);
      else if(vm["type"].as<std::string>() == "gf16")
         process< gf<4,0x13>,vector >(vm["system-file"].as<std::string>(), std::cin, std::cout);
      else if(vm["type"].as<std::string>() == "sigspace")
         process<sigspace,vector>(vm["system-file"].as<std::string>(), std::cin, std::cout);
      else
         {
         std::cerr << "Unrecognized symbol type: " << vm["type"].as<std::string>() << "\n";
         return 1;
         }
      }
   else if(vm["container"].as<std::string>() == "matrix")
      {
      using libbase::matrix;
      using libbase::gf;
      using libcomm::sigspace;
      if(vm["type"].as<std::string>() == "bool")
         process<bool,matrix>(vm["system-file"].as<std::string>(), std::cin, std::cout);
      else if(vm["type"].as<std::string>() == "gf2")
         process< gf<1,0x3>,matrix >(vm["system-file"].as<std::string>(), std::cin, std::cout);
      else if(vm["type"].as<std::string>() == "gf4")
         process< gf<2,0x7>,matrix >(vm["system-file"].as<std::string>(), std::cin, std::cout);
      else if(vm["type"].as<std::string>() == "gf8")
         process< gf<3,0xB>,matrix >(vm["system-file"].as<std::string>(), std::cin, std::cout);
      else if(vm["type"].as<std::string>() == "gf16")
         process< gf<4,0x13>,matrix >(vm["system-file"].as<std::string>(), std::cin, std::cout);
      else if(vm["type"].as<std::string>() == "sigspace")
         process<sigspace,matrix>(vm["system-file"].as<std::string>(), std::cin, std::cout);
      else
         {
         std::cerr << "Unrecognized symbol type: " << vm["type"].as<std::string>() << "\n";
         return 1;
         }
      }
   else
      {
      std::cerr << "Unrecognized container type: " << vm["container"].as<std::string>() << "\n";
      return 1;
      }

   return 0;
   }

}; // end namespace

int main(int argc, char *argv[])
   {
   return csencode::main(argc, argv);
   }
