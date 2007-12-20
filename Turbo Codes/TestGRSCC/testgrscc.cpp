/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "rscc.h"
#include "grscc.h"
#include "gf.h"
#include <iostream>

using std::cout;
using std::cerr;

using libbase::vector;
using libbase::matrix;

using libbase::gf;
using libcomm::grscc;
using libbase::bitfield;
using libcomm::rscc;

// Define types for binary and for GF(2^4): m(x) = 1 { 0011 }
typedef gf<1,0x3>  GF2;
typedef gf<4,0x13> GF16;

matrix< vector<GF16> > GetGeneratorGF16()
   {
   // Create generator matrix for a R=1/3 code
   matrix< vector<GF16> > gen(1,3);
   // 1 + D + a^4 D^2
   gen(0,0).init(3);
   gen(0,0)(2) = "0001";
   gen(0,0)(1) = "0001";
   gen(0,0)(0) = "0011";
   // 1 + a D + a^4 D^2
   gen(0,1).init(3);
   gen(0,1)(2) = "0001";
   gen(0,1)(1) = "0010";
   gen(0,1)(0) = "0011";
   // 1 + a^2 D + a^9 D^2
   gen(0,2).init(3);
   gen(0,2)(2) = "0001";
   gen(0,2)(1) = "0100";
   gen(0,2)(0) = "1010";
   return gen;
   }

matrix< vector<GF2> > GetGeneratorGF2()
   {
   // Create generator matrix for a R=1/2 code
   matrix< vector<GF2> > gen(1,2);
   // 1 + D + D^2
   gen(0,0).init(3);
   gen(0,0)(2) = "1";
   gen(0,0)(1) = "1";
   gen(0,0)(0) = "1";
   // 1 + D^2
   gen(0,1).init(3);
   gen(0,1)(2) = "1";
   gen(0,1)(1) = "0";
   gen(0,1)(0) = "1";
   return gen;
   }

matrix<bitfield> GetGeneratorBinary()
   {
   // Create generator matrix for a R=1/2 code
   matrix<bitfield> gen(1,2);
   // 1 + D + D^2
   gen(0,0) = "111";
   // 1 + D^2
   gen(0,1) = "101";
   return gen;
   }

int main(int argc, char *argv[])
   {
   // Create RSC code from generator matrix for R=1/3, nu=3, GF(16)
   grscc<GF16> cc(GetGeneratorGF16());
   // Show code description
   cout << "Code description:\n";
   cout << cc.description() << "\n";
   // Show code serialization
   cout << "Code serialization: [" << &cc << "]\n";

   // Compute, display, and compare the state table for an RSC with G = [111,101]
   // between the classic rscc and the grscc using the degenerate binary field
   rscc cc_old(GetGeneratorBinary());
   cout << "Classic Code description:\n";
   cout << cc_old.description() << "\n";

   grscc<GF2> cc_new(GetGeneratorGF2());
   cout << "New Code description:\n";
   cout << cc_new.description() << "\n";

   cout << "PS\tIn\tNS\tOut\n";
   for(int q=0; q<cc_old.num_states(); q++)
      for(int i=0; i<cc_old.num_inputs(); i++)
         {
         cc_old.reset(q);
         const int out = cc_old.step(i);
         const int ns = cc_old.state();
         cc_new.reset(q);
         const int n_ps = cc_new.state();
         const int n_out = cc_new.step(i);
         const int n_ns = cc_new.state();
         cout << q << '-' << n_ps << '\t';
         cout << i << '\t';
         cout << ns << '-' << n_ns << '\t';
         cout << out << '-' << n_out << '\n';
         assert(q == n_ps);
         assert(out == n_out);
         assert(ns == n_ns);
         }

   return 0;
   }
