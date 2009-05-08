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
      libbase::vector<S> received(system->output_block_size());
      received.serialize(sin);
      system->translate(received);
      if(soft)
         {
         libcomm::codec_softout<double>& cdc =
            dynamic_cast< libcomm::codec_softout<double>& >(*system->getcodec());
         libbase::vector< libbase::vector<double> > ptable;
         for(int i=0; i<system->getcodec()->num_iter(); i++)
            cdc.softdecode(ptable);
         for(int i=0; i<ptable.size(); i++)
            ptable(i).serialize(sout);
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

/*!
   \brief   Communication Systems Decoder
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
   if(vm["type"].as<std::string>() == "bool")
      process<bool>(vm["system-file"].as<std::string>(),
         vm["parameter"].as<double>(), vm["soft-out"].as<bool>(),
         std::cin, std::cout);
   else if(vm["type"].as<std::string>() == "gf2")
      process< libbase::gf<1,0x3> >(vm["system-file"].as<std::string>(),
         vm["parameter"].as<double>(), vm["soft-out"].as<bool>(),
         std::cin, std::cout);
   else if(vm["type"].as<std::string>() == "gf4")
      process< libbase::gf<2,0x7> >(vm["system-file"].as<std::string>(),
         vm["parameter"].as<double>(), vm["soft-out"].as<bool>(),
         std::cin, std::cout);
   else if(vm["type"].as<std::string>() == "gf8")
      process< libbase::gf<3,0xB> >(vm["system-file"].as<std::string>(),
         vm["parameter"].as<double>(), vm["soft-out"].as<bool>(),
         std::cin, std::cout);
   else if(vm["type"].as<std::string>() == "gf16")
      process< libbase::gf<4,0x13> >(vm["system-file"].as<std::string>(),
         vm["parameter"].as<double>(), vm["soft-out"].as<bool>(),
         std::cin, std::cout);
   else if(vm["type"].as<std::string>() == "sigspace")
      process<libcomm::sigspace>(vm["system-file"].as<std::string>(),
         vm["parameter"].as<double>(), vm["soft-out"].as<bool>(),
         std::cin, std::cout);
   else
      {
      std::cerr << "Unrecognized symbol type: " << vm["type"].as<std::string>() << "\n";
      return 1;
      }

   return 0;
   }
