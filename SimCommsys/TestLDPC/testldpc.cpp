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

int main(int argc, char *argv[])
   {
   if (argc == 1)
      {
      cerr << "\nPlease provide a path to a file";
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
