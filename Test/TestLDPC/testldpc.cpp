/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "serializer_libcomm.h"
#include <iostream>
#include <string>
#include "codec/ldpc.h"
#include "matrix.h"
#include "vector.h"
#include "gf.h"
#include "bitfield.h"
#include "codec/turbo.h"
#include "sumprodalg/impl/sum_prod_alg_gdl.h"
#include "linear_code_utils.h"
#include "randgen.h"

#include <boost/program_options.hpp>

#include <fstream>

using std::cerr;
using std::cout;
using std::string;
using libcomm::ldpc;
using std::ifstream;
using std::ofstream;
using libcomm::sum_prod_alg_gdl;

namespace testldpc {

using libbase::matrix;
using libbase::gf;
using libbase::randgen;
using libbase::linear_code_utils;

/*!
 \brief Test program for LDPC class
 \author  Steve Wesemeyer
 */

typedef libbase::vector<double> array1d_t;
typedef libbase::vector<int> array1i_t;
typedef libbase::vector<array1i_t> array1vi_t;
typedef libbase::vector<array1d_t> array1vd_t;

void compute_conv(array1d_t& conv_out, int pos1, int pos2)
   {
   if ((pos2 - pos1) == 1)
      {
      double tmp1 = conv_out(pos1);
      double tmp2 = conv_out(pos2);
      conv_out(pos1) = tmp1 + tmp2;
      conv_out(pos2) = tmp1 - tmp2;
      }
   else
      {
      int midpoint = pos1 + (pos2 - pos1 + 1) / 2;
      compute_conv(conv_out, pos1, midpoint - 1);
      compute_conv(conv_out, midpoint, pos2);
      pos2 = midpoint;
      for (int loop1 = pos1; loop1 < midpoint; loop1++)
         {
         double tmp1 = conv_out(loop1);
         double tmp2 = conv_out(pos2);
         conv_out(loop1) = tmp1 + tmp2;
         conv_out(pos2) = tmp1 - tmp2;
         pos2++;
         }
      }
   }

void compute_dual()
   {
   matrix<gf<1, 0x3> > test;
   test.init(5, 31);
   int tmp;
   for (int i = 1; i < 32; i++)
      {
      tmp = i;
      for (int shift = 0; shift < 5; shift++)
         {
         test(shift, i - 1) = tmp % 2;
         tmp = tmp >> 1;
         }
      }

   cout << "the parity check matrix for the (31,26) Hamming code looks like:"
         << std::endl;
   test.serialize(cout, ' ');
   cout << std::endl;
   //compute the minimum weight codewords
   array1i_t info_sym;
   array1i_t code_word;
   for (int i = 0; i < 31; i++)
      {
      info_sym = test.extractcol(i);
      linear_code_utils<gf<1, 0x3>, double>::encode_cw(test, info_sym,
            code_word);
      cout << "codeword " << i + 1 << ": ";
      code_word.serialize(cout, ' ');
      cout << " weight= " << code_word.sum() << std::endl;
      }

   /*
    matrix<gf<1, 0x3> > dual;
    array1i_t systematic_perm;
    linear_code_utils<gf<1, 0x3> , double>::compute_dual_code(test, dual, systematic_perm);
    cout << "the dual matrix is given by:" << std::endl;
    dual.serialize(cout, ' ');
    */
   }

void test_ra_code()
   {
   //needed for serialisation
   const libcomm::serializer_libcomm my_serializer_libcomm;

   //we are going to look at a RA code over GF(8) with [

   typedef libbase::gf<3, 0xB> gf_t;
   string tmp_dir = "/user/cscssst/sw0024/";
   string filename = tmp_dir + "ra_code_paras.txt";

   libcomm::sysrepacc<double> sys_ra;

   ifstream sin(filename.c_str());
   sys_ra.serialize(sin);
   cout << sys_ra.description() << std::endl;
   int k = sys_ra.input_block_size();
   cout << "input block size: " << k << std::endl;
   int n = sys_ra.output_block_size();
   cout << "output block size: " << n << std::endl;

   //init the matrix which is going to hold the generator matrix
   libbase::matrix<gf_t> gen_matrix;
   gen_matrix.init(k, n);

   //init the matrix which is going to hold the parity check matrix
   libbase::matrix<gf_t> pc_matrix;
   pc_matrix.init(n - k, n);

   //interleaved versions
   libbase::matrix<gf_t> pc_matrix_inter, pc_matrix_inter_ref, gen_matrix_inter;
   pc_matrix_inter.init(n - k, n);
   pc_matrix_inter_ref.init(n - k, n);
   gen_matrix_inter.init(k, n);

   int stepsize = n / k;

   libbase::vector<int> input, output, output2;

   input.init(k);
   output.init(n);
   output2.init(n);
   for (int i = 0; i < k; i++)
      {
      input *= 0; //reset to zero
      input(i) = 1;
      sys_ra.encode(input, output);
      cout << "encoding: " << input << std::endl;
      cout << "yields: " << output << std::endl;
      cout << "looking at just the input bits" << std::endl;
      int counter = 0;
      for (int j = 0; j < n; j = j + stepsize)
         {
         cout << output(j) << "\t";
         gen_matrix(i, counter) = output(j);
         counter++;
         }
      cout << std::endl;
      for (int j = 0; j < n; j++)
         {
         if ((j % stepsize) != 0)
            {
            gen_matrix(i, counter) = output(j);
            counter++;
            }
         }
      }
   cout << "we have generated the following gen_matrix:" << std::endl;
   cout << gen_matrix;
   libbase::vector<int> systematic_perm;
   linear_code_utils<gf_t, double>::compute_dual_code(gen_matrix, pc_matrix,
         systematic_perm);
   cout << "the dual matrix is given by:" << std::endl;
   pc_matrix.serialize(cout, ' ');
   cout << "the permutation is given by: " << std::endl;
   cout << systematic_perm << std::endl;

   //we now re-arrange the columns of the parity check matrix so that
   //it represents the interleaved turbo code again.
   int info_counter = 0;
   int par_counter = k;
   for (int i = 0; i < n; i++)
      {
      if ((i % stepsize) == 0)
         {
         pc_matrix_inter.insertcol(pc_matrix.extractcol(info_counter), i);
         info_counter++;
         }
      else
         {
         pc_matrix_inter.insertcol(pc_matrix.extractcol(par_counter), i);
         par_counter++;
         }
      }
   linear_code_utils<gf_t, double>::compute_dual_code(pc_matrix_inter,
         gen_matrix_inter, systematic_perm);
   cout << "the interleaved gen matrix is given by:" << std::endl;
   gen_matrix_inter.serialize(cout, ' ');
   cout << "the permutation is given by: " << std::endl;
   cout << systematic_perm << std::endl;

   //Sanity check this with a basis of the code space

   for (int i = 0; i < k; i++)
      {
      input *= 0; //reset to zero
      input(i) = 1;
      sys_ra.encode(input, output);
      linear_code_utils<gf_t, double>::encode_cw(gen_matrix_inter, input,
            output2);
      if (output.isequalto(output2))
         {
         cout << i << "\t: test failed" << std::endl;
         }
      else
         {
         cout << i << "\t: test passed" << std::endl;
         }
      }
   //test
   pc_matrix_inter_ref = pc_matrix_inter.reduce_to_ref();

   //use the parity check matrix to generate an LDPC code which we will serialise out

   ldpc<gf_t, double> ldpc_sys_ra(pc_matrix_inter_ref, 100);
   //ldpc<gf_t, double> ldpc_turbo(pc_matrix_inter, 100);
   std::ostringstream oss;
   oss << tmp_dir << "ldpc_sys_ra_" << n << "x" << k << "_ser.txt";

   string outfile_ser = oss.str();
   ofstream sout_ser(outfile_ser.c_str());
   ldpc_sys_ra.serialize(sout_ser);

   }

void test_cc_code()
   {
   //needed for serialisation
   const libcomm::serializer_libcomm my_serializer_libcomm;

   //we are going to look at a TC over GF(8) with dim[100,500]

   typedef libbase::gf<3, 0xB> gf_t;
   string tmp_dir = "/user/cscssst/sw0024/";
   string filename = tmp_dir + "turbocode_paras.txt";

   libcomm::turbo<double, double> turbo;

   ifstream sin(filename.c_str());
   turbo.serialize(sin);
   cout << turbo.description() << std::endl;
   int k = turbo.input_block_size();
   cout << "input block size: " << k << std::endl;
   int n = turbo.output_block_size();
   cout << "output block size: " << n << std::endl;

   //init the matrix which is going to hold the generator matrix
   libbase::matrix<gf_t> gen_matrix;
   gen_matrix.init(k, n);

   //init the matrix which is going to hold the parity check matrix
   libbase::matrix<gf_t> pc_matrix;
   pc_matrix.init(n - k, n);

   //interleaved versions
   libbase::matrix<gf_t> pc_matrix_inter, pc_matrix_inter_ref, gen_matrix_inter;
   pc_matrix_inter.init(n - k, n);
   pc_matrix_inter_ref.init(n - k, n);
   gen_matrix_inter.init(k, n);

   int stepsize = n / k;

   libbase::vector<int> input, output, output2;

   input.init(k);
   output.init(n);
   output2.init(n);
   for (int i = 0; i < k; i++)
      {
      input *= 0; //reset to zero
      input(i) = 1;
      turbo.encode(input, output);
      cout << "encoding: " << input << std::endl;
      cout << "yields: " << output << std::endl;
      cout << "looking at just the input bits" << std::endl;
      int counter = 0;
      for (int j = 0; j < n; j = j + stepsize)
         {
         cout << output(j) << "\t";
         gen_matrix(i, counter) = output(j);
         counter++;
         }
      cout << std::endl;
      for (int j = 0; j < n; j++)
         {
         if ((j % stepsize) != 0)
            {
            gen_matrix(i, counter) = output(j);
            counter++;
            }
         }
      }
   cout << "we have generated the following gen_matrix:" << std::endl;
   cout << gen_matrix;
   libbase::vector<int> systematic_perm;
   linear_code_utils<gf_t, double>::compute_dual_code(gen_matrix, pc_matrix,
         systematic_perm);
   cout << "the dual matrix is given by:" << std::endl;
   pc_matrix.serialize(cout, ' ');
   cout << "the permutation is given by: " << std::endl;
   cout << systematic_perm << std::endl;

   //we now re-arrange the columns of the parity check matrix so that
   //it represents the interleaved turbo code again.
   int info_counter = 0;
   int par_counter = k;
   for (int i = 0; i < n; i++)
      {
      if ((i % stepsize) == 0)
         {
         pc_matrix_inter.insertcol(pc_matrix.extractcol(info_counter), i);
         info_counter++;
         }
      else
         {
         pc_matrix_inter.insertcol(pc_matrix.extractcol(par_counter), i);
         par_counter++;
         }
      }
   linear_code_utils<gf_t, double>::compute_dual_code(pc_matrix_inter,
         gen_matrix_inter, systematic_perm);
   cout << "the interleaved gen matrix is given by:" << std::endl;
   gen_matrix_inter.serialize(cout, ' ');
   cout << "the permutation is given by: " << std::endl;
   cout << systematic_perm << std::endl;

   //Sanity check this with a basis of the code space

   for (int i = 0; i < k; i++)
      {
      input *= 0; //reset to zero
      input(i) = 1;
      turbo.encode(input, output);
      linear_code_utils<gf_t, double>::encode_cw(gen_matrix_inter, input,
            output2);
      if (output.isequalto(output2))
         {
         cout << i << "\t: test failed" << std::endl;
         }
      else
         {
         cout << i << "\t: test passed" << std::endl;
         }
      }
   //test
   pc_matrix_inter_ref = pc_matrix_inter.reduce_to_ref();

   //use the parity check matrix to generate an LDPC code which we will serialise out

   ldpc<gf_t, double> ldpc_turbo(pc_matrix_inter_ref, 100);
   //ldpc<gf_t, double> ldpc_turbo(pc_matrix_inter, 100);
   std::ostringstream oss;
   oss << tmp_dir << "ldpc_turbo_" << n << "x" << k << "_ser.txt";

   string outfile_ser = oss.str();
   ofstream sout_ser(outfile_ser.c_str());
   ldpc_turbo.serialize(sout_ser);

   }

template <class GF>
void process(bool serialized)
   {
   // read the alist LDPC code
   ldpc<GF, double> codec;
   codec.read_alist(std::cin);
   std::cerr << "Codec: " << codec.description() << std::endl;

   if (serialized)
      {
      // write it in serialised format
      std::cout << &codec;
      }
   else
      {
      // write it in alist format
      codec.write_alist(std::cout);
      }
   }

int main(int argc, char *argv[])
   {
   namespace po = boost::program_options;

   // Set up user parameters
   po::options_description desc("Allowed options");
   desc.add_options()("help", "print this help message");
   desc.add_options()("serialized,s", po::bool_switch(),
         "convert alist to serialized format");
   desc.add_options()("alist,a", po::bool_switch(),
         "convert alist to alist format");
   desc.add_options()("type,t", po::value<std::string>()->default_value("gf2"),
         "LDPC alphabet");
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   po::notify(vm);

   // read switch parameters
   const bool s = vm["serialized"].as<bool>();
   const bool a = vm["alist"].as<bool>();

   // Validate user parameters
   if (vm.count("help") || (!s && !a) || (s && a))
      {
      cout << desc << std::endl;
      return 1;
      }

   // Shorthand access for parameters
   const std::string type = vm["type"].as<std::string>();

   using libbase::gf;
   if (type == "gf2")
      process<gf<1, 0x3> >(s);
   else if (type == "gf4")
      process<gf<2, 0x7> >(s);
   else if (type == "gf8")
      process<gf<3, 0xB> >(s);
   else if (type == "gf16")
      process<gf<4, 0x13> >(s);
   else if (type == "gf32")
      process<gf<5, 0x25> >(s);
   else if (type == "gf64")
      process<gf<6, 0x43> >(s);
   else if (type == "gf128")
      process<gf<7, 0x89> >(s);
   else if (type == "gf256")
      process<gf<8, 0x11D> >(s);
   else if (type == "gf512")
      process<gf<9, 0x211> >(s);
   else if (type == "gf1024")
      process<gf<10, 0x409> >(s);
   else
      {
      std::cerr << "Unrecognized symbol type: " << type << std::endl;
      return 1;
      }
   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return testldpc::main(argc, argv);
   }
