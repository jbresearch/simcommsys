#include "serializer_libcomm.h"
#include "commsys.h"
#include "commsys_fulliter.h"
#include "codec/codec_softout.h"
#include "timer.h"

#include <boost/program_options.hpp>
#include <iostream>

namespace csdecode {

template <class S, template <class > class C>
void process(const std::string& fname, double p, bool softin, bool softout,
      std::istream& sin = std::cin, std::ostream& sout = std::cout)
   {
   // define types
   typedef libbase::vector<double> array1d_t;
   typedef libcomm::codec_softout<C> codec_so;

   // Communication system
   libcomm::commsys<S, C> *system = libcomm::loadfromfile<
         libcomm::commsys<S, C> >(fname);
   std::cerr << system->description() << "\n";
   // Set channel parameter
   system->getchan()->set_parameter(p);
   // Initialize system
   libbase::randgen r;
   r.seed(0);
   system->seedfrom(r);

   // Repeat until end of stream
   while (!sin.eof())
      {
      if (softin)
         {
         C<array1d_t> ptable_in(system->output_block_size());
         ptable_in.serialize(sin);
         system->softreceive_path(ptable_in);
         }
      else
         {
         C<S> received(system->output_block_size());
         received.serialize(sin);
         system->receive_path(received);
         }
      libbase::eatwhite(sin);
      if (softout)
         {
         codec_so& cdc = dynamic_cast<codec_so&> (*system->getcodec());
         C<array1d_t> ptable_out;
         for (int i = 0; i < system->num_iter(); i++)
            cdc.softdecode(ptable_out);
         ptable_out.serialize(sout);
         }
      else
         {
         C<int> decoded;
         for (int i = 0; i < system->num_iter(); i++)
            system->decode(decoded);
         decoded.serialize(sout, '\n');
         }
      }
   }

/*!
 * \brief   Communication Systems Decoder
 * \author  Johann Briffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

int main(int argc, char *argv[])
   {
   libbase::timer tmain("Main timer");

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
   desc.add_options()("parameter,p", po::value<double>(), "channel parameter");
   desc.add_options()("soft-in,s", po::bool_switch(), "enable soft input");
   desc.add_options()("soft-out,o", po::bool_switch(), "enable soft output");
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   po::notify(vm);

   // Validate user parameters
   if (vm.count("help") || vm.count("system-file") == 0
         || vm.count("parameter") == 0)
      {
      std::cerr << desc << "\n";
      return 1;
      }
   // Shorthand access for parameters
   const std::string container = vm["container"].as<std::string> ();
   const std::string type = vm["type"].as<std::string> ();
   const std::string filename = vm["system-file"].as<std::string> ();
   const double parameter = vm["parameter"].as<double> ();
   const bool softin = vm["soft-in"].as<bool> ();
   const bool softout = vm["soft-out"].as<bool> ();

   // Main process
   if (container == "vector")
      {
      using libbase::vector;
      using libbase::gf;
      using libcomm::sigspace;
      if (type == "bool")
         process<bool, vector> (filename, parameter, softin, softout);
      else if (type == "gf2")
         process<gf<1, 0x3> , vector> (filename, parameter, softin, softout);
      else if (type == "gf4")
         process<gf<2, 0x7> , vector> (filename, parameter, softin, softout);
      else if (type == "gf8")
         process<gf<3, 0xB> , vector> (filename, parameter, softin, softout);
      else if (type == "gf16")
         process<gf<4, 0x13> , vector> (filename, parameter, softin, softout);
      else if (type == "gf32")
         process<gf<5, 0x25> , vector> (filename, parameter, softin, softout);
      else if (type == "gf64")
         process<gf<6, 0x43> , vector> (filename, parameter, softin, softout);
      else if (type == "gf128")
         process<gf<7, 0x89> , vector> (filename, parameter, softin, softout);
      else if (type == "gf256")
         process<gf<8, 0x11D> , vector> (filename, parameter, softin, softout);
      else if (type == "gf512")
         process<gf<9, 0x211> , vector> (filename, parameter, softin, softout);
      else if (type == "gf1024")
         process<gf<10, 0x409> , vector> (filename, parameter, softin, softout);
      else if (type == "sigspace")
         process<sigspace, vector> (filename, parameter, softin, softout);
      else
         {
         std::cerr << "Unrecognized symbol type: " << type << "\n";
         return 1;
         }
      }
   else if (container == "matrix")
      {
      using libbase::matrix;
      using libbase::gf;
      using libcomm::sigspace;
      if (type == "bool")
         process<bool, matrix> (filename, parameter, softin, softout);
      else if (type == "gf2")
         process<gf<1, 0x3> , matrix> (filename, parameter, softin, softout);
      else if (type == "gf4")
         process<gf<2, 0x7> , matrix> (filename, parameter, softin, softout);
      else if (type == "gf8")
         process<gf<3, 0xB> , matrix> (filename, parameter, softin, softout);
      else if (type == "gf16")
         process<gf<4, 0x13> , matrix> (filename, parameter, softin, softout);
      else if (type == "gf32")
         process<gf<5, 0x25> , matrix> (filename, parameter, softin, softout);
      else if (type == "gf64")
         process<gf<6, 0x43> , matrix> (filename, parameter, softin, softout);
      else if (type == "gf128")
         process<gf<7, 0x89> , matrix> (filename, parameter, softin, softout);
      else if (type == "gf256")
         process<gf<8, 0x11D> , matrix> (filename, parameter, softin, softout);
      else if (type == "gf512")
         process<gf<9, 0x211> , matrix> (filename, parameter, softin, softout);
      else if (type == "gf1024")
         process<gf<10, 0x409> , matrix> (filename, parameter, softin, softout);
      else if (type == "sigspace")
         process<sigspace, matrix> (filename, parameter, softin, softout);
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

} // end namespace

int main(int argc, char *argv[])
   {
   return csdecode::main(argc, argv);
   }
