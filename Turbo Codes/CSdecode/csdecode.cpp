#include "serializer_libcomm.h"
#include "commsys.h"
#include "codec_softout.h"
#include "timer.h"

#include <boost/program_options.hpp>
#include <iostream>

namespace csdecode {

template <class S, template<class> class C>
void process(const std::string& fname, double p, bool soft, std::istream& sin=std::cin, std::ostream& sout=std::cout)
   {
   // Communication system
   libcomm::commsys<S,C> *system = libcomm::loadfromfile< libcomm::commsys<S,C> >(fname);
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
      C<S> received(system->output_block_size());
      received.serialize(sin);
      system->receive_path(received);
      if(soft)
         {
         libcomm::codec_softout<C>& cdc =
            dynamic_cast< libcomm::codec_softout<C>& >(*system->getcodec());
         C< libbase::vector<double> > ptable;
         for(int i=0; i<system->getcodec()->num_iter(); i++)
            cdc.softdecode(ptable);
         ptable.serialize(sout);
         }
      else
         {
         C<int> decoded;
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
      ("container,c", po::value<std::string>()->default_value("vector"),
         "input/output container type")
      ("parameter,p", po::value<double>(),
         "channel parameter")
      ("soft-out,s", po::bool_switch(),
         "enable soft output")
      ;
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   po::notify(vm);

   // Validate user parameters
   if(vm.count("help") || \
      vm.count("system-file")==0 || \
      vm.count("parameter")==0)
      {
      std::cerr << desc << "\n";
      return 1;
      }
   // Shorthand access for parameters
   const std::string container = vm["container"].as<std::string>();
   const std::string type = vm["type"].as<std::string>();
   const std::string filename = vm["system-file"].as<std::string>();
   const double parameter = vm["parameter"].as<double>();
   const bool softout = vm["soft-out"].as<bool>();

   // Main process
   if(container == "vector")
      {
      using libbase::vector;
      using libbase::gf;
      using libcomm::sigspace;
      if(type == "bool")
         process<bool,vector>(filename, parameter, softout);
      else if(type == "gf2")
         process< gf<1,0x3>,vector >(filename, parameter, softout);
      else if(type == "gf4")
         process< gf<2,0x7>,vector >(filename, parameter, softout);
      else if(type == "gf8")
         process< gf<3,0xB>,vector >(filename, parameter, softout);
      else if(type == "gf16")
         process< gf<4,0x13>,vector >(filename, parameter, softout);
      else if(type == "sigspace")
         process<sigspace,vector>(filename, parameter, softout);
      else
         {
         std::cerr << "Unrecognized symbol type: " << type << "\n";
         return 1;
         }
      }
   else if(container == "matrix")
      {
      using libbase::matrix;
      using libbase::gf;
      using libcomm::sigspace;
      if(type == "bool")
         process<bool,matrix>(filename, parameter, softout);
      else if(type == "gf2")
         process< gf<1,0x3>,matrix >(filename, parameter, softout);
      else if(type == "gf4")
         process< gf<2,0x7>,matrix >(filename, parameter, softout);
      else if(type == "gf8")
         process< gf<3,0xB>,matrix >(filename, parameter, softout);
      else if(type == "gf16")
         process< gf<4,0x13>,matrix >(filename, parameter, softout);
      else if(type == "sigspace")
         process<sigspace,matrix>(filename, parameter, softout);
      else
         {
         std::cerr << "Unrecognized symbol type: " << type << "\n";
         return 1;
         }
      }
   else
      {
      std::cerr << "Unrecognized container type: " << container << "\n";
      return 1;
      }

   return 0;
   }

}; // end namespace

int main(int argc, char *argv[])
   {
   return csdecode::main(argc, argv);
   }
