/*!
 * \file
 * \brief Boost unit tests for quantum_gaussian_source and cvqkd_protocol
 *
 * Copyright (c) 2025 Aaron Abela
 */

#define BOOST_TEST_MODULE GaussianSourceTest
#include <boost/test/included/unit_test.hpp>


#include "source/quantum_gaussian_source.h"
#include "qkd/observable/position_observable.h"
#include "qkd/observable/momentum_observable.h"
#include "qkd/qkd_protocol/cvqkd_protocol.h"
#include "qkd/quantum_channel/gaussian_quantum_channel.h"
#include "qkd/quantum_channel/identity_quantum_channel.h"
#include "truerand.h"

#include "qkd_commsys.h"
#include "qkd/quantum_state.h"
#include "qkd/qkd_protocol.h"
#include "qkd/observable.h"
#include "qkd/quantum_channel.h"
#include "source/quantum_gaussian_source.h"
#include "random.h"
#include "vector.h"

#include <memory>
#include <vector>

#include <iostream>
#include <cmath>

using namespace libcomm;
using namespace libbase;

// Force-link the registration translation units by constructing each type and calling its *output* serialize.
// This ensures that the shelpers run and register the types.
static void force_link_qkd_types()
{
  std::ostringstream oss;

  {
    // identity_quantum_channel TU
    libcomm::identity_quantum_channel obj;
    static_cast<const libcomm::identity_quantum_channel&>(obj).serialize(oss);
  }
  {
    // gaussian_quantum_channel TU
    libcomm::gaussian_quantum_channel obj;
    static_cast<const libcomm::gaussian_quantum_channel&>(obj).serialize(oss);
  }
  {
    // cvqkd_protocol TU
    libcomm::cvqkd_protocol obj;
    static_cast<const libcomm::cvqkd_protocol&>(obj).serialize(oss);
  }
}

BOOST_AUTO_TEST_CASE(test_qkd_commsys_object_up_until_measurement)
{
   std::cout << "\n*****Boost Test Case *****\n";
   std::cout << "\nTesting QKD Commsys up until Measurement\n";

   // Ensure registrars are linked & run
   force_link_qkd_types();

   // 1) Build a config that matches qkd_commsys::serialize(std::istream&)
   std::stringstream cfg;
   cfg <<
      "# Version\n"
      "1\n"
      "# Frame size (# of quantum states in a frame)\n"
      "5000\n"
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
      "500\n"
      "# Shot Noise Variance N_0\n"
      "1\n"
      "# Electric Noise v_el\n"
      "0.041\n"
      "# Detector Efficiency eta\n"
      "0.606\n";

   // 2) Default-construct the system and load the config
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

   // 4) Print System Parameters of the QKD Commsys Object
   std::cout << "\n" << sys.description() << "\n\n";

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

   // Setting modulation variance VA in the gaussian quantum channel of Bob
   sys.set_VA(*src);

   // Generates a sequence of coherent states which is the input to the fullcycle method in qkd_commsys.h
   libbase::vector<gaussian_state> source = src->generate_sequence(libbase::size_type<libbase::vector>(framesize));

   // Initialise final_key
   libbase::vector<bool> final_key;

   /* Calling fullcylce method from qkd_commsys.h for a single frame*/
   final_key = sys.fullcycle(source);

   // Prints Final Secret Key
   std::cout << "\nFinal Secret Key [size=" << final_key.size() << "]: [";
   for (int i = 0; i < final_key.size(); ++i) {
      if (i) std::cout << ", ";
      std::cout << final_key(i);
   }
   std::cout << "]\n\n";

}