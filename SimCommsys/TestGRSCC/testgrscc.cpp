#include "fsm/rscc.h"
#include "fsm/grscc.h"
#include "gf.h"
#include <iostream>

namespace testgrscc {

using std::cout;
using std::cerr;

using libbase::vector;
using libbase::matrix;

using libbase::gf;
using libcomm::grscc;
using libbase::bitfield;
using libcomm::rscc;
using libcomm::fsm;

// Define types for binary and for GF(2^4): m(x) = 1 { 0011 }
typedef gf<1, 0x3> GF2;
typedef gf<3, 0xB> GF8;
typedef gf<4, 0x13> GF16;

matrix<vector<GF16> > GetGeneratorGF16()
   {
   // Create generator matrix for a R=1/3 code
   matrix<vector<GF16> > gen(1, 3);
   // 1 + D + a^4 D^2
   gen(0, 0).init(3);
   gen(0, 0)(0) = "0001";
   gen(0, 0)(1) = "0001";
   gen(0, 0)(2) = "0011";
   // 1 + a D + a^4 D^2
   gen(0, 1).init(3);
   gen(0, 1)(0) = "0001";
   gen(0, 1)(1) = "0010";
   gen(0, 1)(2) = "0011";
   // 1 + a^2 D + a^9 D^2
   gen(0, 2).init(3);
   gen(0, 2)(0) = "0001";
   gen(0, 2)(1) = "0100";
   gen(0, 2)(2) = "1010";
   return gen;
   }

matrix<vector<GF8> > GetGeneratorGF8()
   {
   // Create generator matrix for a R=1/2 code
   matrix<vector<GF8> > gen(1, 2);
   // 1 + D + a^4 D^2
   gen(0, 0).init(3);
   gen(0, 0)(0) = "001";
   gen(0, 0)(1) = "001";
   gen(0, 0)(2) = "110";
   // 1 + a D + a^4 D^2
   gen(0, 1).init(3);
   gen(0, 1)(0) = "001";
   gen(0, 1)(1) = "010";
   gen(0, 1)(2) = "110";
   return gen;
   }

matrix<vector<GF2> > GetGeneratorGF2()
   {
   // Create generator matrix for a R=1/2 code
   matrix<vector<GF2> > gen(1, 2);
   // Feedback: 1 + D + D^3 (1011 = 13)
   gen(0, 0).init(4);
   gen(0, 0)(0) = "1";
   gen(0, 0)(1) = "1";
   gen(0, 0)(2) = "0";
   gen(0, 0)(3) = "1";
   // Output: 1 + D + D^2 + D^3 (1111 = 17)
   gen(0, 1).init(4);
   gen(0, 1)(0) = "1";
   gen(0, 1)(1) = "1";
   gen(0, 1)(2) = "1";
   gen(0, 1)(3) = "1";
   return gen;
   }

matrix<bitfield> GetGeneratorBinary()
   {
   // Create generator matrix for a R=1/2 code
   matrix<bitfield> gen(1, 2);
   // Feedback: 1 + D + D^3 (1011 = 13)
   gen(0, 0) = bitfield("1011");
   // Output: 1 + D + D^2 + D^3 (1111 = 17)
   gen(0, 1) = bitfield("1111");
   return gen;
   }

void TestCreation()
   {
   cout << "\nTest code creation:\n";
   // Create RSC code from generator matrix for R=1/3, nu=2, GF(16)
   grscc<GF16> cc(GetGeneratorGF16());
   // Show code description
   cout << "Code description:\n";
   cout << cc.description() << "\n";
   // Show code serialization
   cout << "Code serialization: [" << &cc << "]\n";
   }

void CompareCode(fsm& enc, const int ns[], const int out[])
   {
   cout << enc.description() << "\n";

   cout << "PS\tIn\tNS\tOut\n";
   for (int ps = 0, k = 0; ps < enc.num_states(); ps++)
      for (int i = 0; i < enc.num_inputs(); i++, k++)
         {
         // reset encoder and verify the state is correctly set
         enc.reset(enc.convert_state(ps));
         const int n_ps = enc.convert_state(enc.state());
         cout << ps << '-' << n_ps << '\t';
         assert(ps == n_ps);
         // prepare required input
         vector<int> ip = enc.convert_input(i);
         cout << i << '\t';
         // feed input and determine output and next state
         const int n_out = enc.convert_output(enc.step(ip));
         const int n_ns = enc.convert_state(enc.state());
         cout << ns[k] << '-' << n_ns << '\t';
         cout << out[k] << '-' << n_out << '\n';
         assert(out[k] == n_out);
         assert(ns[k] == n_ns);
         }
   }

void CompareCodes()
   {
   cout << "\nTest comparison of code with classic one:\n";
   /* Consider a RSC with G = [1011,1111]
    * PS        In      NS      Out
    * 000       0       000     00
    * 000       1       001     11
    * 001       0       011     10
    * 001       1       010     01
    * 010       0       100     10
    * 010       1       101     01
    * 011       0       111     00
    * 011       1       110     11
    * 100       0       001     10
    * 100       1       000     01
    * 101       0       010     00
    * 101       1       011     11
    * 110       0       101     00
    * 110       1       100     11
    * 111       0       110     10
    * 111       1       111     01
    */
   const int ns[] = {0, 1, 3, 2, 4, 5, 7, 6, 1, 0, 2, 3, 5, 4, 6, 7};
   const int out[] = {0, 3, 2, 1, 2, 1, 0, 3, 2, 1, 0, 3, 0, 3, 2, 1};

   // Compute, display, and compare the state table

   cout << "Generalized Code:\n";
   grscc<GF2> cc_new(GetGeneratorGF2());
   CompareCode(cc_new, ns, out);

   cout << "Binary Code:\n";
   rscc cc_old(GetGeneratorBinary());
   CompareCode(cc_old, ns, out);
   }

void TestCirculation()
   {
   cout << "\nTest code circulation:\n";
   // Create RSC code from generator matrix for R=1/2, nu=2, GF(8)
   grscc<GF8> cc(GetGeneratorGF8());
   // Show code description
   cout << "Code description:\n";
   cout << cc.description() << "\n";
   // Encode a short all-zero sequence
   //cc.reset();
   //int ip = 0;
   //for(int i=0; i<16; i++)
   //   cc.advance(ip);
   // Call circulation reset
   //cc.resetcircular();
   // Compute and display circulation state correspondence table
   for (int S = 0; S < cc.num_states(); S++)
      cout << '\t' << S;
   for (int N = 1; N < 7; N++)
      {
      cout << '\n' << N;
      for (int S = 0; S < cc.num_states(); S++)
         {
         cc.resetcircular(cc.convert_state(S), N);
         cout << '\t' << cc.state();
         }
      }
   cout << '\n';
   }

/*!
 * \brief   Test program for GRSCC class
 * \author  Johann Briffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

int main(int argc, char *argv[])
   {
   TestCreation();
   CompareCodes();
   TestCirculation();
   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return testgrscc::main(argc, argv);
   }
