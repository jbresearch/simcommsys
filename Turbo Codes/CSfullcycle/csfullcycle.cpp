#include "serializer_libcomm.h"
#include "commsys.h"
#include "codec_softout.h"
#include "timer.h"

#include <boost/program_options.hpp>
#include <iostream>

namespace csfullcycle {

template <class S, template<class> class C>
void process(const std::string& fname, double p, bool soft, std::istream& sin, std::ostream& sout)
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
   for(int j=0; !sin.eof(); j++)
      {
      std::cerr << "Processing block " << j << ".";
      C<int> source(system->input_block_size());
      source.serialize(sin);
      std::cerr << ".";
      C<S> transmitted = system->encode(source);
      std::cerr << ".";
      C<S> received;
      system->getchan()->transmit(transmitted, received);
      std::cerr << ".";
      system->translate(received);
      std::cerr << ".";
      if(soft)
         {
         libcomm::codec_softout<double,C>& cdc =
            dynamic_cast< libcomm::codec_softout<double,C>& >(*system->getcodec());
         C< libbase::vector<double> > ptable;
         for(int i=0; i<system->getcodec()->num_iter(); i++)
            cdc.softdecode(ptable);
         std::cerr << ".";
         ptable.serialize(sout);
         }
      else
         {
         C<int> decoded;
         for(int i=0; i<system->getcodec()->num_iter(); i++)
            system->getcodec()->decode(decoded);
         std::cerr << ".";
         decoded.serialize(sout, '\n');
         }
      libbase::eatwhite(sin);
      std::cerr << "done.\n";
      }
   }

/*!
   \brief   Communication Systems Encoder-Transmit-Decoder Cycle
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   \todo Add '-t' option as in csencode/decode
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
   using libbase::vector;
   process<bool,vector>(vm["system-file"].as<std::string>(),
      vm["parameter"].as<double>(), vm["soft-out"].as<bool>(),
      std::cin, std::cout);

   return 0;
   }

}; // end namespace

int main(int argc, char *argv[])
   {
   return csfullcycle::main(argc, argv);
   }
