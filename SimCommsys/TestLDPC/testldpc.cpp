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
 * 
 * \section svn Version Control
 * - $Id$
 */

#include <iostream>
#include <string>
#include "codec/ldpc.h"
#include "matrix.h"
#include "vector.h"
#include "gf.h"
#include "sumprodalg/impl/sum_prod_alg_gdl.h"
#include "linear_code_utils.h"
#include "randgen.h"
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

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$
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

   cout << "the parity check matrix for the (31,26) Hamming code looks like:" << std::endl;
   test.serialize(cout, ' ');
   cout<<std::endl;
   //compute the minimum weight codewords
   array1i_t info_sym;
   array1i_t code_word;
   for (int i = 0; i < 31; i++)
      {
      info_sym = test.extractcol(i);
      linear_code_utils<gf<1, 0x3> , double>::encode_cw(test,info_sym,code_word);
      cout<<"codeword "<<i+1<<": ";
      code_word.serialize(cout,' ');
      cout<<" weight= "<<code_word.sum()<<std::endl;
      }

   /*
    matrix<gf<1, 0x3> > dual;
    array1i_t systematic_perm;
    linear_code_utils<gf<1, 0x3> , double>::compute_dual_code(test, dual, systematic_perm);
    cout << "the dual matrix is given by:" << std::endl;
    dual.serialize(cout, ' ');
    */
   }

int main(int argc, char *argv[])
   {
   compute_dual();
   return 0;
   if (argc == 1)
      {
      cerr << std::endl << "Please provide a path to a file";
      return -1;
      }
   string in_str(argv[1]);
   string infile = in_str + ".txt";
   string outfile_ser = in_str + "_ser.txt";
   string outfile_al = in_str + "_al.txt";
   ldpc<gf<1, 0x3> , double> ldpc_bin;

   //read the alist LDPC code
   ifstream sin(infile.c_str());
   ldpc_bin.read_alist(sin);

   //write it in serialised format
   ofstream sout_ser(outfile_ser.c_str());
   ldpc_bin.serialize(sout_ser);

   //write it in alist format
   ofstream sout_al(outfile_al.c_str());
   ldpc_bin.write_alist(sout_al);
   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return testldpc::main(argc, argv);
   }
