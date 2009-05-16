#include "gf.h"
#include "bitfield.h"
#include "matrix.h"
#include <iostream>

namespace testgf {

using libbase::gf;
using libbase::bitfield;
using libbase::matrix;

using std::cout;
using std::cerr;
using std::hex;
using std::dec;

/*!
   \brief Exponential table entries for base {03}
   cf. Gladman, "A Specification for Rijndael, the AES Algorithm", 2003, p.5
*/
const int aestable[] = {
   0x01, 0x03, 0x05, 0x0f, 0x11, 0x33, 0x55, 0xff, 0x1a, 0x2e, 0x72, 0x96, 0xa1, 0xf8, 0x13, 0x35,
   0x5f, 0xe1, 0x38, 0x48, 0xd8, 0x73, 0x95, 0xa4, 0xf7, 0x02, 0x06, 0x0a, 0x1e, 0x22, 0x66, 0xaa,
   0xe5, 0x34, 0x5c, 0xe4, 0x37, 0x59, 0xeb, 0x26, 0x6a, 0xbe, 0xd9, 0x70, 0x90, 0xab, 0xe6, 0x31,
   0x53, 0xf5, 0x04, 0x0c, 0x14, 0x3c, 0x44, 0xcc, 0x4f, 0xd1, 0x68, 0xb8, 0xd3, 0x6e, 0xb2, 0xcd,
   0x4c, 0xd4, 0x67, 0xa9, 0xe0, 0x3b, 0x4d, 0xd7, 0x62, 0xa6, 0xf1, 0x08, 0x18, 0x28, 0x78, 0x88,
   0x83, 0x9e, 0xb9, 0xd0, 0x6b, 0xbd, 0xdc, 0x7f, 0x81, 0x98, 0xb3, 0xce, 0x49, 0xdb, 0x76, 0x9a,
   0xb5, 0xc4, 0x57, 0xf9, 0x10, 0x30, 0x50, 0xf0, 0x0b, 0x1d, 0x27, 0x69, 0xbb, 0xd6, 0x61, 0xa3,
   0xfe, 0x19, 0x2b, 0x7d, 0x87, 0x92, 0xad, 0xec, 0x2f, 0x71, 0x93, 0xae, 0xe9, 0x20, 0x60, 0xa0,
   0xfb, 0x16, 0x3a, 0x4e, 0xd2, 0x6d, 0xb7, 0xc2, 0x5d, 0xe7, 0x32, 0x56, 0xfa, 0x15, 0x3f, 0x41,
   0xc3, 0x5e, 0xe2, 0x3d, 0x47, 0xc9, 0x40, 0xc0, 0x5b, 0xed, 0x2c, 0x74, 0x9c, 0xbf, 0xda, 0x75,
   0x9f, 0xba, 0xd5, 0x64, 0xac, 0xef, 0x2a, 0x7e, 0x82, 0x9d, 0xbc, 0xdf, 0x7a, 0x8e, 0x89, 0x80,
   0x9b, 0xb6, 0xc1, 0x58, 0xe8, 0x23, 0x65, 0xaf, 0xea, 0x25, 0x6f, 0xb1, 0xc8, 0x43, 0xc5, 0x54,
   0xfc, 0x1f, 0x21, 0x63, 0xa5, 0xf4, 0x07, 0x09, 0x1b, 0x2d, 0x77, 0x99, 0xb0, 0xcb, 0x46, 0xca,
   0x45, 0xcf, 0x4a, 0xde, 0x79, 0x8b, 0x86, 0x91, 0xa8, 0xe3, 0x3e, 0x42, 0xc6, 0x51, 0xf3, 0x0e,
   0x12, 0x36, 0x5a, 0xee, 0x29, 0x7b, 0x8d, 0x8c, 0x8f, 0x8a, 0x85, 0x94, 0xa7, 0xf2, 0x0d, 0x17,
   0x39, 0x4b, 0xdd, 0x7c, 0x84, 0x97, 0xa2, 0xfd, 0x1c, 0x24, 0x6c, 0xb4, 0xc7, 0x52, 0xf6, 0x01
   };

void TestBinaryField()
   {
   // Create values in the Binary field GF(2): m(x) = 1 { 1 }
   typedef gf<1,0x3> Binary;
   // Compute and display addition & multiplication tables
   cout << "\nBinary Addition table:\n";
   for(int x=0; x<2; x++)
      for(int y=0; y<2; y++)
         cout << Binary(x)+Binary(y) << (y==1 ? '\n' : '\t');
   cout << "\nBinary Multiplication table:\n";
   for(int x=0; x<2; x++)
      for(int y=0; y<2; y++)
         cout << Binary(x)*Binary(y) << (y==1 ? '\n' : '\t');
   }

void TestRijndaelField()
   {
   // Create a value in the Rijndael field GF(2^8): m(x) = 1 { 0001 1011 }
   gf<8,0x11B> E = 1;
   // Compute and display exponential table using {03} as a multiplier
   // using the tabular format in Gladman.
   cout << "\nRijndael GF(2^8) exponentiation table:\n";
   cout << hex;
   for(int x=0; x<16; x++)
      for(int y=0; y<16; y++)
         {
         assert(E == aestable[(x<<4)+y]);
         cout << int(E) << (y==15 ? '\n' : '\t');
         E *= 3;
         }
   cout << dec;
   }

template <int m, int poly>
void ListField()
   {
   // Compute and display exponential table using {2} as a multiplier
   cout << "\nGF(" << m << ",0x" << hex << poly << dec << ") table:\n";
   cout << 0 << '\t' << 0 << '\t' << bitfield(0,m) << '\n';
   gf<m,poly> E = 1;
   for(int x=1; x < gf<m,poly>::elements(); x++)
      {
      cout << x << "\ta" << x-1 << '\t' << bitfield(E,m) << '\n';
      E *= 2;
      }
   }

template <int m, int poly>
void TestMulDiv()
   {
   // Compute and display exponential table using {2} as a multiplier
   cout << "\nGF(" << m << ",0x" << hex << poly << dec << ") multiplication/division:\n";
   cout << "power\tvalue\tinverse\tmul\n";
   gf<m,poly> E = 1;
   for(int x=1; x < gf<m,poly>::elements(); x++)
      {
      cout << "a" << x-1 << '\t' << bitfield(E,m) << '\t' << bitfield(E.inverse(),m) << '\t' << bitfield(E.inverse()*E,m) << '\n';
      E *= 2;
      }
   }

void TestGenPowerGF2()
   {
   cout << "\nBinary generator matrix power sequence:\n";
   // Create values in the Binary field GF(2): m(x) = 1 { 1 }
   typedef gf<1,0x3> Binary;
   // Create generator matrix for DVB-CRSC code:
   matrix<Binary> G(3,3);
   G = 0;
   G(0,0) = 1;
   G(2,0) = 1;
   G(0,1) = 1;
   G(1,2) = 1;
   // Compute and display first 8 powers of G
   for(int i=0; i<8; i++)
      {
      cout << "G^" << i << " = \n";
      pow(G,i).serialize(cout);
      }
   }

void TestGenPowerGF8()
   {
   cout << "\nGF(8) generator matrix power sequence:\n";
   // Create values in the field GF(8): m(x) = 1 { 011 }
   typedef gf<3,0xB> GF8;
   // Create generator matrix:
   matrix<GF8> G(2,2);
   G(0,0) = 1;
   G(1,0) = 6;
   G(0,1) = 1;
   G(1,1) = 0;
   // Compute and display first 16 powers of G
   for(int i=0; i<16; i++)
      {
      cout << "G^" << i << " = \n";
      pow(G,i).serialize(cout);
      }
   }

/*!
   \brief Test program for GF class
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

int main(int argc, char *argv[])
   {
   TestBinaryField();
   TestRijndaelField();
   ListField<2,0x7>();
   ListField<3,0xB>();
   ListField<4,0x13>();
   TestMulDiv<3,0xB>();
   TestGenPowerGF2();
   TestGenPowerGF8();
   return 0;
   }

}; // end namespace

int main(int argc, char *argv[])
   {
   return testgf::main(argc, argv);
   }
