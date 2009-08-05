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
 - $Revision: 2799 $
 - $Date: 2009-08-04 14:43:53 +0100 (Tue, 04 Aug 2009) $
 - $Author: swesemeyer $
 */
/*
 int main(int argc, char *argv[])
 {
 randgen rng;
 rng.seed(123);
 int length_n = 12;
 int max_iter = 10;
 int tmp_pos = 0;
 matrix<gf<1, 0x3> > pchk_matrix;
 matrix<gf<1, 0x3> > gen_matrix;
 libbase::vector<int> perm;
 for (int dim_k = 3; dim_k < 10; dim_k++)
 {
 for (int iter = 0; iter < max_iter; iter++)
 {
 pchk_matrix.init(dim_k, length_n);
 pchk_matrix = gf<1, 0x3> (0);
 //create a random matrix
 for (int row = 0; row < dim_k; row++)
 {
 int num_of_row_entries = 3;
 for (int loop = 0; loop < num_of_row_entries; loop++)
 {
 tmp_pos = int(rng.fval() * (length_n - 0.000001));
 pchk_matrix(row, tmp_pos) = gf<1, 0x3> (1);
 }
 }
 //compute the dual code
 linear_code_utils<gf<1, 0x3> , double>::compute_dual_code(pchk_matrix,
 gen_matrix, perm);
 }
 }
 return 0;
 }
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

/*         int size = probs.size();
 conv_out = probs;
 int index = 1;
 int pos1 = 0;
 int pos2 = 0;
 int convs = size / 2;
 double tmp1;
 double tmp2;
 while         (index < size)
 {
 pos1 = 0;
 for (int loop = 0; loop < convs; loop++)
 {

 pos1++;
 if (pos1 == pos2)
 {
 pos1++;
 }
 }
 index << 1; // *2
 }
 */

int main(int argc, char *argv[])
   {
   /*
   array1d_t convs;
   convs.init(8);
   for (int loop1 = 1; loop1 < 9; loop1++)
      {
      convs(loop1 - 1) = loop1;
      }
   compute_conv(convs, 0, 7);
   convs.serialize(std::cout, ' ');
*/
   if (argc == 1)
      {
      cerr << "\nPlease provide a path to a file";
      return -1;
      }
   string in_str(argv[1]);
   string infile = in_str + ".txt";
   string outfile_ser = in_str + "_ser.txt";
   string outfile_al = in_str + "_al.txt";
   ldpc<gf<4, 0x13> , double> ldpc_bin;

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
