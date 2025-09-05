/*!
 * \file
 * \brief Boost unit tests for quantum_gaussian_source and cvqkd_protocol
 */

#define BOOST_TEST_MODULE GaussianSourceTest
#include <boost/test/included/unit_test.hpp>

#include "qkd/qkd_protocol/cvqkd_protocol.h"
#include "qkd/quantum_channel/gaussian_quantum_channel.h"
#include "qkd/quantum_channel/identity_quantum_channel.h"
#include "source/quantum_gaussian_source.h"
#include "qkd_commsys.h"

#include "serializer.h"
#include "random.h"
#include "vector.h"
#include "gf.h"
#include "codec/ldpc.h"

#include <iostream>
#include <sstream>
#include <memory>
#include <vector>
#include <algorithm>

using namespace libcomm;
using namespace libbase;

// Force-link registrars by touching const serialize (runs TU static registration)
static void force_link_qkd_types() {
    std::ostringstream oss;
    { libcomm::identity_quantum_channel i; static_cast<const libcomm::identity_quantum_channel&>(i).serialize(oss); }
    { libcomm::gaussian_quantum_channel g; static_cast<const libcomm::gaussian_quantum_channel&>(g).serialize(oss); }

   // {
   // libcomm::ldpc<libbase::gf2, double> c;
   // // Calling serialize() on the const base triggers vtable usage and
   // // ensures the explicit instantiation + registrar in ldpc.cpp are linked.
   // static_cast<const libcomm::ldpc<libbase::gf2, double>&>(c).serialize(oss);
   // }

    {
    // cvqkd_protocol TU
    libcomm::cvqkd_protocol obj;
    static_cast<const libcomm::cvqkd_protocol&>(obj).serialize(oss);
    }


}

// static void force_link_ldpc() {
//   // Touch serialize() so the TU’s registrar isn’t discarded by the linker.
//   std::ostringstream oss;
//   libcomm::ldpc<libbase::gf2,double> tmp;
//   static_cast<const libcomm::ldpc<libbase::gf2,double>&>(tmp).serialize(oss);
// }

BOOST_AUTO_TEST_CASE(test_qkd_commsys_object_up_until_measurement)
{
    std::cout << "\n*****Boost Test Case *****\n";

   // Ensure registrars are linked & run
   force_link_qkd_types();

   libcomm::ldpc<libbase::gf2,double> codec; // Without this entire qkd_commsys serialization won't work as the codec can't be loaded!!!

   //  force_link_ldpc();

    // Build config: GAUSSIAN for Alice with explicit params (parse-proof)
    std::stringstream cfg;
    cfg <<
        "# Version\n"
        "1\n"
        "# Frame size (# of quantum states in a frame)\n"
        "110000\n"
        "## Alice's channel\n"
        "identity_quantum_channel\n"
        "## Bob's channel\n"
        "gaussian_quantum_channel\n"
        "# Homodyne Detector Efficiency\n"
        "0.606\n"
        "# Mean of the Gaussian Quantum Channel\n"
        "0.0\n"
        "# Transmittance T of the Gaussian Quantum Channel\n"
        "0.302\n"
        "## Postprocessing protocol\n"
        "cvqkd_protocol\n"
        "# Number of samples for Parameter Estimation N_PE\n"
        "11000\n"
        "# Shot Noise Variance N_0\n"
        "1\n"
        "# Electric Noise v_el\n"
        "0.041\n"
        "# Detector Efficiency eta\n"
        "0.606\n"
        "# Smoothing Parameter\n"
        "1e-4\n";
      //   "## Codec\n"
      //   "ldpc<gf2,double>\n"
      //   "# Version\n"
      //   "5\n"
      //   "# SPA type (trad|gdl)\n"
      //   "gdl\n"
      //   "# Number of iterations\n"
      //   "50\n"
      //   "# Clipping method\n"
      //   "zero\n"
      //   "# Value of almostzero\n"
      //   "1e-100\n"
      //   "# Reduce generator matrix to REF? (true|false)\n"
      //   "1\n"
      //   "# Length (n)\n"
      //   "7\n"
      //   "# Dimension (m)\n"
      //   "7\n"
      //   "# Max column weight\n"
      //   "3\n"
      //   "# Max row weight\n"
      //   "3\n"
      //   "# Non-zero values (ones|random|provided)\n"
      //   "ones\n"
      //   "# Column weight vector\n"
      //   "7\n"
      //   "3 3 3 3 3 3 3\n"
      //   "# Row weight vector\n"
      //   "7\n"
      //   "3 3 3 3 3 3 3\n"
      //   "# Non zero positions per col\n"
      //   "3\n"
      //   "1 5 7\n"
      //   "3\n"
      //   "1 2 6\n"
      //   "3\n"
      //   "2 3 7\n"
      //   "3\n"
      //   "1 3 4\n"
      //   "3\n"
      //   "2 4 5\n"
      //   "3\n"
      //   "3 5 6\n"
      //   "3\n"
      //   "4 6 7\n";

   libcomm::qkd_commsys<libcomm::gaussian_state, double, libbase::vector> sys;

   std::cout << "Derived under quantum_channel:\n";
   for (auto& s : libbase::serializer::get_derived_classes("quantum_channel"))
      std::cout << " - " << s << "\n";

   sys.serialize(cfg);   // this populates alice_channel, bob_channel, protocol, framesize

   // Set seed for qkd_commsys object
   randgen rng;
   rng.seed(17);
   sys.seedfrom(rng);

   // 3) Set the CLI params (Bob’s SNR) through qkd_commsys
   // const double SNR_linear = 17.7558; // linear (not dB) excess noise of 0.005
   const double SNR_linear =  17.74018571692195; // for an excess noise of 0.01

   libbase::vector<double> cli;
   cli.init(sys.get_num_params());  // should be 1 when Alice is identity , CLI channel parameters
   cli(0) = SNR_linear;             // index 0 -> Bob's SNR
   sys.set_parameters(cli);

   // 4a) Print System Parameters of the QKD Commsys Object
   std::cout << "\n" << sys.description() << "\n\n";

   // // 4b) Print Codec details of the CV-QKD protocol under qkd_commsys.h.
   // std::cout << "\nPrinting codec description from qkd_commsys.h" << std::endl;
   // std::cout << sys.codec_description();


   // 5) Verify CLI parameters of Quantum Channel of Bob
   auto back = sys.get_parameters();
   std::cout << "Print (SNR) CLI parameter of Bob's Quantum Channel " << back << std::endl;

      // 6) Create Gaussian Quantum Source
   std::stringstream ss_src;
   ss_src << "# Mean of Q_Mean\n"
         << "0.0\n"
         << "# Stddev of Q_Mean\n"
         << "4.30116\n"
         << "# Mean of P_Mean\n"
         << "0.0\n"
         << "# Stddev of P_Mean\n"
         << "4.30116\n"
         << "# Stddev of Q\n"
         << "1.0\n"
         << "# Stddev of P\n"
         << "1.0\n";

   // Build source using the same pattern as gaussian_quantum_channel
   std::unique_ptr<libbase::serializable> s_ptr = libcomm::quantum_gaussian_source::create(ss_src);
   auto* src = dynamic_cast<libcomm::quantum_gaussian_source*>(s_ptr.get());
   BOOST_REQUIRE(src != nullptr);

   randgen r;
   r.seed(2602);
   // r.seed(7896); variance was 16.9 (should be a bit closer to 18.5)
   src->seedfrom(r);

   double VA = src->get_VA();
   std::cout << "\n Checking Modulation Variance of Source = " << VA << std::endl;

   // Gets the number of coherent states generated for a single frame from the qkd_commsys object.
   int framesize = sys.input_block_size();
   std::cout << "Number of generated quantum states from Alice = " << framesize << std::endl;

   // Setting modulation variance VA in the gaussian quantum channel of Bob
   sys.set_VA(*src);

   // Generates a sequence of coherent states which is the input to the fullcycle method in qkd_commsys.h
   libbase::vector<gaussian_state> source = src->generate_sequence(libbase::size_type<libbase::vector>(framesize));

   // Initialise final_key
   libbase::vector<bool> final_key;

   // auto qkd_commsys_codec = sys.getcodec();
   // std::cout << "\nPrint description of the qkd_commsys codec: " << qkd_commsys_codec->description();


   /* Calling fullcylce method from qkd_commsys.h for a single frame*/
   final_key = sys.fullcycle(source);

   // // Prints Final Secret Key
   // std::cout << "\nFinal Secret Key [size=" << final_key.size() << "]: [";
   // for (int i = 0; i < final_key.size(); ++i) {
   //    if (i) std::cout << ", ";
   //    std::cout << final_key(i);
   // }
   // std::cout << "]\n\n";
}
