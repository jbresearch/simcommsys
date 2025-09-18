/*!
 * \file
 *
 * Copyright (c) 2025 Aaron Abela
 * \brief Boost unit tests for quantum_gaussian_source and cvqkd_protocol
 */

#define BOOST_TEST_MODULE GaussianSourceTest
#include <boost/test/included/unit_test.hpp>

#include <iostream>
#include <sstream>
#include <memory>
#include <vector>
#include <algorithm>

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

   template <typename T>
   void print_vector(const std::string& title, const libbase::vector<T>& vec)
   {
      std::cout << "\n" << title << std::endl;
      for (int i = 0; i < vec.size(); ++i)
      {
            std::cout << vec(i) << "\t";
      }
      std::cout << std::endl;
   }




BOOST_AUTO_TEST_CASE(test_qkd_commsys_object_up_until_measurement)
{
    std::cout << "\n*****Boost Test Case *****\n";

   // Ensure registrars are linked & run
   force_link_qkd_types();

   // libcomm::ldpc<libbase::gf2,double> codec; // Without this entire qkd_commsys serialization won't work as the codec can't be loaded!!!

    // Build config: GAUSSIAN for Alice with explicit params (parse-proof) // Using N = 110K states and N_PE was previously a set parameter.
   //  std::stringstream cfg;
   //  cfg <<
   //      "# Version\n"
   //      "1\n"
   //      "# Frame size (# of quantum states in a frame)\n"
   //      "110000\n"
   //      "## Alice's channel\n"
   //      "identity_quantum_channel\n"
   //      "## Bob's channel\n"
   //      "gaussian_quantum_channel\n"
   //      "# Homodyne Detector Efficiency\n"
   //      "0.606\n"
   //      "# Mean of the Gaussian Quantum Channel\n"
   //      "0.0\n"
   //      "# Transmittance T of the Gaussian Quantum Channel\n"
   //      "0.302\n"
   //      "## Postprocessing protocol\n"
   //      "cvqkd_protocol\n"
   //      "# Number of samples for Parameter Estimation N_PE\n"
   //      "11000\n"
   //      "# Shot Noise Variance N_0\n"
   //      "1\n"
   //      "# Electric Noise v_el\n"
   //      "0.041\n"
   //      "# Detector Efficiency eta\n"
   //      "0.606\n"
   //      "# Smoothing Parameter\n"
   //      "1e-4\n";
   //    //   "### Codec\n"
   //    //   "ldpc<gf2,double>\n"
   //    //   "# Version\n"
   //    //   "5\n"
   //    //   "# SPA type (trad|gdl)\n"
   //    //   "gdl\n"
   //    //   "# Number of iterations\n"
   //    //   "50\n"
   //    //   "# Clipping method\n"
   //    //   "zero\n"
   //    //   "# Value of almostzero\n"
   //    //   "1e-100\n"
   //    //   "# Reduce generator matrix to REF? (true|false)\n"
   //    //   "1\n"
   //    //   "# Length (n)\n"
   //    //   "7\n"
   //    //   "# Dimension (m)\n"
   //    //   "7\n"
   //    //   "# Max column weight\n"
   //    //   "3\n"
   //    //   "# Max row weight\n"
   //    //   "3\n"
   //    //   "# Non-zero values (ones|random|provided)\n"
   //    //   "ones\n"
   //    //   "# Column weight vector\n"
   //    //   "7\n"
   //    //   "3 3 3 3 3 3 3\n"
   //    //   "# Row weight vector\n"
   //    //   "7\n"
   //    //   "3 3 3 3 3 3 3\n"
   //    //   "# Non zero positions per col\n"
   //    //   "3\n"
   //    //   "1 5 7\n"
   //    //   "3\n"
   //    //   "1 2 6\n"
   //    //   "3\n"
   //    //   "2 3 7\n"
   //    //   "3\n"
   //    //   "1 3 4\n"
   //    //   "3\n"
   //    //   "2 4 5\n"
   //    //   "3\n"
   //    //   "3 5 6\n"
   //    //   "3\n"
   //    //   "4 6 7\n";


   // Build config: GAUSSIAN for Alice with explicit params (parse-proof) // Using N = 14 states and N_PE is no longer a serialized parameter. It is calculated N_PE = framesize - n (from codec)
   //  std::stringstream cfg;
   //  cfg <<
   //      "# Version\n"
   //      "1\n"
   //      "# Frame size (# of quantum states in a frame)\n"
   //      "14\n"
   //      "## Alice's channel\n"
   //      "identity_quantum_channel\n"
   //      "## Bob's channel\n"
   //      "gaussian_quantum_channel\n"
   //      "# Homodyne Detector Efficiency\n"
   //      "0.606\n"
   //      "# Mean of the Gaussian Quantum Channel\n"
   //      "0.0\n"
   //      "# Transmittance T of the Gaussian Quantum Channel\n"
   //      "0.302\n"
   //      "## Postprocessing protocol\n"
   //      "cvqkd_protocol\n"
   //      "# Shot Noise Variance N_0\n"
   //      "1\n"
   //      "# Electric Noise v_el\n"
   //      "0.041\n"
   //      "# Detector Efficiency eta\n"
   //      "0.606\n"
   //      "# Smoothing Parameter\n"
   //      "1e-4\n";

    std::stringstream cfg;
    cfg <<
        "# Version\n"
        "1\n"
        "# Frame size (# of quantum states in a frame)\n"
        "30\n"
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
        "# Shot Noise Variance N_0\n"
        "1\n"
        "# Electric Noise v_el\n"
        "0.041\n"
        "# Detector Efficiency eta\n"
        "0.606\n"
        "# Smoothing Parameter\n"
        "1e-4\n"
        "# Alphabet size\n"
        "2\n";

   libcomm::qkd_commsys<libcomm::gaussian_state, double, libbase::vector> sys;

   std::cout << "Derived under quantum_channel:\n";
   for (auto& s : libbase::serializer::get_derived_classes("quantum_channel"))
      std::cout << " - " << s << "\n";

   sys.serialize(cfg);   // this populates alice_channel, bob_channel, protocol, framesize

   // Set seed for qkd_commsys object
   libbase::randgen rng;
   rng.seed(17);
   // rng.seed(26);
   // rng.seed(7);

   /*

   With this seed:
   rng.seed(17);

   and a
   randgen r; // for src and vector s
   r.seed(2602); // for src and  vector s

   I had the following:
   - Generated vector s of Bob = [1 1 1]
   - Generated vector C (after encoding s) = [1       1       1       0       0       1       0]
   - Generated vector M = -0.324052       -0.172396       0.0642522       0.840933        -0.233945       -0.0788091      0.305013
   - Generated probability table:
   (Prints probability table from cv-qkdprotocol.cpp:)
   t=0 : 0.000037, 0.361188
   t=1 : 0.396687, 0.059911
   t=2 : 0.392867, 0.335163
   t=3 : 0.201802, 0.000000
   t=4 : 0.388728, 0.004239
   t=5 : 0.005572, 0.000231
   t=6 : 0.049393, 0.112772
   - decoded sequence s_hat = [1 1 1]
   */

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

   libbase::randgen r;
   r.seed(2602);
   // r.seed(26);
   // r.seed(7);

   // r.seed(7896); variance was 16.9 (should be a bit closer to 18.5)
   src->seedfrom(r);

   double VA = src->get_VA();
   std::cout << "\n Checking Modulation Variance of Source = " << VA << std::endl;

   // Gets the number of coherent states generated for a single frame from the qkd_commsys object.
   int framesize = sys.input_block_size();
   std::cout << "Number of generated quantum states from Alice = " << framesize << std::endl;

   // // Test 1: Parameters of LDPC codec with n=7, k=3 and m=4. For this test I used a framesize=14 and 50% for PE.
   // std::stringstream ss;
   //          ss <<
   //          "# Version\n"
   //          "5\n"
   //          "# SPA type (trad|gdl)\n"
   //          "gdl\n"
   //          "# Number of iterations\n"
   //          "50\n"
   //          "# Clipping method\n"
   //          "zero\n"
   //          "# Value of almostzero\n"
   //          "1e-100\n"
   //          "# Reduce generator matrix to REF? (true|false)\n"
   //          "1\n"
   //          "# Length (n)\n"
   //          "7\n"
   //          "# Dimension (m)\n"
   //          "7\n"
   //          "# Max column weight\n"
   //          "3\n"
   //          "# Max row weight\n"
   //          "3\n"
   //          "# Non-zero values (ones|random|provided)\n"
   //          "ones\n"
   //          "# Column weight vector\n"
   //          "7\n"
   //          "3 3 3 3 3 3 3\n"
   //          "# Row weight vector\n"
   //          "7\n"
   //          "3 3 3 3 3 3 3\n"
   //          "# Non zero positions per col\n"
   //          "3\n"
   //          "1 5 7\n"
   //          "3\n"
   //          "1 2 6\n"
   //          "3\n"
   //          "2 3 7\n"
   //          "3\n"
   //          "1 3 4\n"
   //          "3\n"
   //          "2 4 5\n"
   //          "3\n"
   //          "3 5 6\n"
   //          "3\n"
   //          "4 6 7\n";

   // Test 2: Parameters of LDPC codec with n=96, k=48 and m=48. For this test I used a framesize=126 and 23.8% for PE.
   // std::stringstream ss;
   //          ss <<
   //          "# Version\n"
   //          "5\n"
   //          "# SPA type (trad|gdl)\n"
   //          "gdl\n"
   //          "# Number of iterations\n"
   //          "200\n"
   //          "# Clipping method\n"
   //          "zero\n"
   //          "# Value of almostzero\n"
   //          "1e-100\n"
   //          "# Reduce generator matrix to REF? (true|false)\n"
   //          "0\n"
   //          "# Length (n)\n"
   //          "96\n"
   //          "# Dimension (m)\n"
   //          "48\n"
   //          "# Max column weight\n"
   //          "3\n"
   //          "# Max row weight\n"
   //          "6\n"
   //          "# Non-zero values (ones|random|provided)\n"
   //          "ones\n"
   //          "# Column weight vector\n"
   //          "96\n"
   //          "3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3\n"
   //          "# Row weight vector\n"
   //          "48\n"
   //          "6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6\n"
   //          "# Non zero positions per col\n"
   //          "3\n"
   //          "47 4 21\n"
   //          "3\n"
   //          "33 38 31\n"
   //          "3\n"
   //          "11 1 33\n"
   //          "3\n"
   //          "3 48 37\n"
   //          "3\n"
   //          "42 9 36\n"
   //          "3\n"
   //          "17 22 7\n"
   //          "3\n"
   //          "48 15 13\n"
   //          "3\n"
   //          "40 28 47\n"
   //          "3\n"
   //          "22 42 5\n"
   //          "3\n"
   //          "28 33 30\n"
   //          "3\n"
   //          "27 18 19\n"
   //          "3\n"
   //          "2 34 10\n"
   //          "3\n"
   //          "38 41 27\n"
   //          "3\n"
   //          "18 7 32\n"
   //          "3\n"
   //          "16 32 45\n"
   //          "3\n"
   //          "26 24 1\n"
   //          "3\n"
   //          "25 16 22\n"
   //          "3\n"
   //          "35 25 34\n"
   //          "3\n"
   //          "37 2 11\n"
   //          "3\n"
   //          "21 3 39\n"
   //          "3\n"
   //          "34 21 28\n"
   //          "3\n"
   //          "12 13 6\n"
   //          "3\n"
   //          "1 39 38\n"
   //          "3\n"
   //          "9 8 12\n"
   //          "3\n"
   //          "44 12 48\n"
   //          "3\n"
   //          "29 14 9\n"
   //          "3\n"
   //          "31 29 26\n"
   //          "3\n"
   //          "5 46 14\n"
   //          "3\n"
   //          "36 6 24\n"
   //          "3\n"
   //          "46 23 3\n"
   //          "3\n"
   //          "45 30 4\n"
   //          "3\n"
   //          "24 11 8\n"
   //          "3\n"
   //          "23 10 42\n"
   //          "3\n"
   //          "7 35 43\n"
   //          "3\n"
   //          "32 19 41\n"
   //          "3\n"
   //          "19 20 25\n"
   //          "3\n"
   //          "15 47 46\n"
   //          "3\n"
   //          "39 31 2\n"
   //          "3\n"
   //          "13 43 20\n"
   //          "3\n"
   //          "43 40 15\n"
   //          "3\n"
   //          "8 5 35\n"
   //          "3\n"
   //          "4 26 44\n"
   //          "3\n"
   //          "6 37 17\n"
   //          "3\n"
   //          "10 45 18\n"
   //          "3\n"
   //          "20 27 29\n"
   //          "3\n"
   //          "30 17 16\n"
   //          "3\n"
   //          "41 36 23\n"
   //          "3\n"
   //          "14 44 40\n"
   //          "3\n"
   //          "7 31 42\n"
   //          "3\n"
   //          "25 23 21\n"
   //          "3\n"
   //          "22 34 41\n"
   //          "3\n"
   //          "42 3 19\n"
   //          "3\n"
   //          "40 35 27\n"
   //          "3\n"
   //          "21 19 17\n"
   //          "3\n"
   //          "4 8 28\n"
   //          "3\n"
   //          "35 45 31\n"
   //          "3\n"
   //          "2 28 32\n"
   //          "3\n"
   //          "37 30 9\n"
   //          "3\n"
   //          "38 40 30\n"
   //          "3\n"
   //          "34 36 13\n"
   //          "3\n"
   //          "33 46 10\n"
   //          "3\n"
   //          "32 12 40\n"
   //          "3\n"
   //          "18 41 11\n"
   //          "3\n"
   //          "17 1 2\n"
   //          "3\n"
   //          "45 39 29\n"
   //          "3\n"
   //          "9 48 4\n"
   //          "3\n"
   //          "47 11 34\n"
   //          "3\n"
   //          "19 29 24\n"
   //          "3\n"
   //          "44 17 5\n"
   //          "3\n"
   //          "15 2 3\n"
   //          "3\n"
   //          "16 21 33\n"
   //          "3\n"
   //          "11 20 44\n"
   //          "3\n"
   //          "20 9 47\n"
   //          "3\n"
   //          "23 47 38\n"
   //          "3\n"
   //          "24 16 12\n"
   //          "3\n"
   //          "41 24 37\n"
   //          "3\n"
   //          "39 5 43\n"
   //          "3\n"
   //          "6 43 23\n"
   //          "3\n"
   //          "31 10 16\n"
   //          "3\n"
   //          "48 33 35\n"
   //          "3\n"
   //          "28 18 48\n"
   //          "3\n"
   //          "8 42 18\n"
   //          "3\n"
   //          "36 32 8\n"
   //          "3\n"
   //          "14 6 25\n"
   //          "3\n"
   //          "29 15 36\n"
   //          "3\n"
   //          "46 38 26\n"
   //          "3\n"
   //          "5 4 6\n"
   //          "3\n"
   //          "27 44 22\n"
   //          "3\n"
   //          "26 22 45\n"
   //          "3\n"
   //          "43 27 1\n"
   //          "3\n"
   //          "10 25 39\n"
   //          "3\n"
   //          "12 14 7\n"
   //          "3\n"
   //          "13 7 46\n"
   //          "3\n"
   //          "30 13 14\n"
   //          "3\n"
   //          "3 26 20\n"
   //          "3\n"
   //          "1 37 15";

   // Test 3: Parameters of LDPC codec with n=15, k=6, m=10. For this test I used a framesize=30 and 50% for PE.
   std::stringstream ss;
            ss <<
            "# Version\n"
            "5\n"
            "# SPA type (trad|gdl)\n"
            "gdl\n"
            "# Number of iterations\n"
            "100\n"
            "# Clipping method\n"
            "zero\n"
            "# Value of almostzero\n"
            "1e-100\n"
            "# Reduce generator matrix to REF? (true|false)\n"
            "0\n"
            "# Length (n)\n"
            "15\n"
            "# Dimension (m)\n"
            "10\n"
            "# Max column weight\n"
            "2\n"
            "# Max row weight\n"
            "3\n"
            "# Non-zero values (ones|random|provided)\n"
            "ones\n"
            "# Column weight vector\n"
            "15\n"
            "2 2 2 2 2 2 2 2 2 2 2 2 2 2 2\n"
            "# Row weight vector\n"
            "10\n"
            "3 3 3 3 3 3 3 3 3 3\n"
            "# Non zero positions per col\n"
            "2\n"
            "1 2\n"
            "2\n"
            "2 3\n"
            "2\n"
            "3 4\n"
            "2\n"
            "4 5\n"
            "2\n"
            "1 5\n"
            "2\n"
            "1 6\n"
            "2\n"
            "2 8\n"
            "2\n"
            "3 10\n"
            "2\n"
            "4 7\n"
            "2\n"
            "5 9\n"
            "2\n"
            "6 7\n"
            "2\n"
            "7 8\n"
            "2\n"
            "8 9\n"
            "2\n"
            "9 10\n"
            "2\n"
            "6 10";

   // Test 4: Parameters of LDPC codec with framesize = 12. N_PE = 50% and n=6, m = 4, k = , R = . Taken from Noel's testqkdldpc.cpp.
   // std::stringstream ss;
   // ss << "# Version\n"
   //         << "5\n"
   //         << "# SPA type (trad|gdl)\n"
   //         << "gdl\n"
   //         << "# Number of iterations\n"
   //         << "100\n"
   //         << "# Clipping method\n"
   //         << "zero\n"
   //         << "# Value of almostzero\n"
   //         << "1e-100\n"
   //         << "# Reduce generator matrix to REF? (true|false)\n"
   //         << "0\n"
   //         << "# Length (n)\n"
   //         << "6\n"
   //         << "# Dimension (m)\n"
   //         << "4\n"
   //         << "# Max column weight\n"
   //         << "3\n"
   //         << "# Max row weight\n"
   //         << "3\n"
   //         << "# Non-zero values (ones|random|provided)\n"
   //         << "ones\n"
   //         << "# Column weight vector\n"
   //         << "6\n"
   //         << "2 3 1 1 1 1\n"
   //         << "# Row weight vector\n"
   //         << "4\n"
   //         << "2 3 2 2\n"
   //         << "# Non zero positions per col\n"
   //         << "2\n"
   //         << "2 3\n"
   //         << "3\n"
   //         << "1 2 4\n"
   //         << "1\n"
   //         << "1\n"
   //         << "1\n"
   //         << "2\n"
   //         << "1\n"
   //         << "3\n"
   //         << "1\n"
   //         << "4\n";

   // Test 5: https://www.inference.org.uk/mackay/codes/EN/C/120.64.3.111. 120.64.3.111 (N=120,K=56,M=64,R= 0.4667) with framesize= 240 and N_PE = 120 -> 50%

   // std::stringstream ss;
   // ss << "# Version\n"
   // << "5\n"
   // << "# SPA type (trad|gdl)\n"
   // << "gdl\n"
   // << "# Number of iterations\n"
   // << "100\n"
   // << "# Clipping method\n"
   // << "zero\n"
   // << "# Value of almostzero\n"
   // << "1e-100\n"
   // << "# Reduce generator matrix to REF? (true|false)\n"
   // << "0\n"
   // << "# Length (n)\n"
   // << "120\n"
   // << "# Dimension (m)\n"
   // << "64\n"
   // << "# Max column weight\n"
   // << "3\n"
   // << "# Max row weight\n"
   // << "6\n"
   // << "# Non-zero values (ones|random|provided)\n"
   // << "ones\n"
   // << "# Column weight vector\n"
   // << "120\n"
   // << "3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3\n"
   // << "# Row weight vector\n"
   // << "64\n"
   // << "6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5\n"
   // << "# Non zero positions per col\n"
   // << "3\n" << "25 53 60\n"
   // << "3\n" << "57 41 35\n"
   // << "3\n" << "11 13 29\n"
   // << "3\n" << "64 60 5\n"
   // << "3\n" << "1 14 45\n"
   // << "3\n" << "52 29 49\n"
   // << "3\n" << "12 15 30\n"
   // << "3\n" << "7 22 27\n"
   // << "3\n" << "36 33 22\n"
   // << "3\n" << "40 32 19\n"
   // << "3\n" << "6 12 56\n"
   // << "3\n" << "26 21 50\n"
   // << "3\n" << "54 36 40\n"
   // << "3\n" << "55 6 9\n"
   // << "3\n" << "34 26 13\n"
   // << "3\n" << "23 16 46\n"
   // << "3\n" << "45 10 31\n"
   // << "3\n" << "43 32 51\n"
   // << "3\n" << "59 43 57\n"
   // << "3\n" << "34 18 2\n"
   // << "3\n" << "4 3 32\n"
   // << "3\n" << "26 61 3\n"
   // << "3\n" << "37 38 24\n"
   // << "3\n" << "40 22 28\n"
   // << "3\n" << "56 62 27\n"
   // << "3\n" << "15 10 6\n"
   // << "3\n" << "21 39 18\n"
   // << "3\n" << "35 64 17\n"
   // << "3\n" << "4 44 39\n"
   // << "3\n" << "58 16 11\n"
   // << "3\n" << "8 24 31\n"
   // << "3\n" << "36 12 48\n"
   // << "3\n" << "4 47 25\n"
   // << "3\n" << "13 63 39\n"
   // << "3\n" << "17 41 3\n"
   // << "3\n" << "20 29 31\n"
   // << "3\n" << "49 9 48\n"
   // << "3\n" << "21 14 23\n"
   // << "3\n" << "62 47 30\n"
   // << "3\n" << "15 7 24\n"
   // << "3\n" << "19 42 37\n"
   // << "3\n" << "5 59 8\n"
   // << "3\n" << "10 23 1\n"
   // << "3\n" << "28 25 30\n"
   // << "3\n" << "20 8 2\n"
   // << "3\n" << "34 55 38\n"
   // << "3\n" << "14 11 37\n"
   // << "3\n" << "50 1 46\n"
   // << "3\n" << "17 18 5\n"
   // << "3\n" << "52 61 27\n"
   // << "3\n" << "2 44 19\n"
   // << "3\n" << "28 58 63\n"
   // << "3\n" << "20 54 7\n"
   // << "3\n" << "35 38 51\n"
   // << "3\n" << "33 9 53\n"
   // << "3\n" << "16 33 42\n"
   // << "3\n" << "64 24 29\n"
   // << "3\n" << "16 9 35\n"
   // << "3\n" << "56 57 19\n"
   // << "3\n" << "21 45 20\n"
   // << "3\n" << "41 55 63\n"
   // << "3\n" << "31 46 51\n"
   // << "3\n" << "36 29 23\n"
   // << "3\n" << "11 35 53\n"
   // << "3\n" << "40 4 1\n"
   // << "3\n" << "62 33 17\n"
   // << "3\n" << "6 31 13\n"
   // << "3\n" << "12 57 37\n"
   // << "3\n" << "43 17 20\n"
   // << "3\n" << "18 1 63\n"
   // << "3\n" << "25 32 5\n"
   // << "3\n" << "13 60 47\n"
   // << "3\n" << "61 55 32\n"
   // << "3\n" << "7 8 33\n"
   // << "3\n" << "60 52 37\n"
   // << "3\n" << "33 25 14\n"
   // << "3\n" << "34 30 61\n"
   // << "3\n" << "18 62 42\n"
   // << "3\n" << "6 11 3\n"
   // << "3\n" << "26 54 9\n"
   // << "3\n" << "42 11 34\n"
   // << "3\n" << "2 22 21\n"
   // << "3\n" << "22 37 49\n"
   // << "3\n" << "59 2 56\n"
   // << "3\n" << "17 10 8\n"
   // << "3\n" << "64 13 16\n"
   // << "3\n" << "39 58 26\n"
   // << "3\n" << "44 15 64\n"
   // << "3\n" << "51 6 34\n"
   // << "3\n" << "1 61 9\n"
   // << "3\n" << "52 12 55\n"
   // << "3\n" << "41 2 5\n"
   // << "3\n" << "3 35 27\n"
   // << "3\n" << "22 18 15\n"
   // << "3\n" << "29 40 44\n"
   // << "3\n" << "49 27 53\n"
   // << "3\n" << "3 43 58\n"
   // << "3\n" << "43 45 24\n"
   // << "3\n" << "32 48 62\n"
   // << "3\n" << "40 25 38\n"
   // << "3\n" << "31 39 30\n"
   // << "3\n" << "46 8 19\n"
   // << "3\n" << "5 48 28\n"
   // << "3\n" << "36 10 49\n"
   // << "3\n" << "48 27 44\n"
   // << "3\n" << "46 26 28\n"
   // << "3\n" << "23 45 51\n"
   // << "3\n" << "4 21 57\n"
   // << "3\n" << "50 47 54\n"
   // << "3\n" << "4 28 54\n"
   // << "3\n" << "63 47 56\n"
   // << "3\n" << "58 14 53\n"
   // << "3\n" << "42 38 10\n"
   // << "3\n" << "60 39 59\n"
   // << "3\n" << "23 38 7\n"
   // << "3\n" << "7 12 50\n"
   // << "3\n" << "36 15 52\n"
   // << "3\n" << "16 20 24\n"
   // << "3\n" << "50 30 59\n"
   // << "3\n" << "14 19 41\n";

   // Gets input k bits from codec of the CV-QKD protocol.
   int k = sys.get_codec_input_bits_k();
   std::cout << "Input bits k of codec of CV-QKD protocol from qkd_commsys/ size of vector s =  " << k << "\n"; // just to test that k is correct. This is also added in simulator.

   // Generate vector s from k as done in qkd_commsys simulator.h
   libbase::vector<bool> vector_s(k);
   for (int i = 0; i < k; ++i)
   {
      vector_s(i) = (r.ival(2) != 0);
   }

   std::cout << "Generated vector s from Testgaussiancvqkd.h: " << std::endl;
   for (int i = 0; i<vector_s.size();++i)
   {
      std::cout << vector_s(i) << "\t";
   }
   std::cout << std::endl;

   sys.set_bob_vector(vector_s);

   // Setting modulation variance VA in the gaussian quantum channel of Bob
   sys.set_VA(*src);

   // Generates a sequence of coherent states which is the input to the fullcycle method in qkd_commsys.h
   libbase::vector<libcomm::gaussian_state> source = src->generate_sequence(libbase::size_type<libbase::vector>(framesize));

   // Initialise final_key
   libbase::vector<bool> final_key;

   /* Calling fullcylce method from qkd_commsys.h for a single frame*/
   auto [key_KA, key_KB]  = sys.fullcycle(source);


   std::cout << "\n Size of Final Secret Key KA: "<< key_KA.size() << std::endl;

//    // // Prints Final Secret Key
//    // std::cout << "\nFinal Secret Key [size=" << final_key.size() << "]: [";
//    // // for (int i = 0; i < final_key.size(); ++i) {
//    // //    if (i) std::cout << ", ";
//    // //    std::cout << final_key(i);
//    // // }
//    // std::cout << "]\n\n";
}
