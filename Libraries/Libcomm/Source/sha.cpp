/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "sha.h"

#include <sstream>

namespace libcomm {

// Const values

const libbase::int32u sha::K[] = { 0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xca62c1d6 };

// Construction/Destruction

sha::sha()
   {
   // reset chaining variables
   m_hash.init(5);
   m_hash = 0;
   }

// Digest-specific functions

void sha::derived_reset()
   {
   // reset chaining variables
   m_hash.init(5);
   m_hash(0) = 0x67452301;
   m_hash(1) = 0xefcdab89;
   m_hash(2) = 0x98badcfe;
   m_hash(3) = 0x10325476;
   m_hash(4) = 0xc3d2e1f0;
   }

void sha::process_block(const libbase::vector<libbase::int32u>& M)
   {
   // create expanded message block
   libbase::vector<libbase::int32u> W;
   expand(M, W);
   // copy variables
   libbase::vector<libbase::int32u> hash = m_hash;
   // main loop
   for(int t=0; t<80; t++)
      {
      const libbase::int32u temp = cshift(hash(0),5) + f(t,hash(1),hash(2),hash(3)) + hash(4) + W(t) + K[t/20];
      hash(4) = hash(3);
      hash(3) = hash(2);
      hash(2) = cshift(hash(1), 30);
      hash(1) = hash(0);
      hash(0) = temp;
      //trace << "SHA: step " << t << "\t" << hex << hash(0) << " " << hash(1) << " " << hash(2) << " " << hash(3) << " " << hash(4) << dec << "\n";
      }
   // add back variables
   m_hash += hash;
   }

// SHA nonlinear function implementations

libbase::int32u sha::f(const int t, const libbase::int32u X, const libbase::int32u Y, const libbase::int32u Z)
   {
   assert(t<80);
   switch(t/20)
      {
      case 0:
         return (X & Y) | ((~X) & Z);
      case 1:
         return X ^ Y ^ Z;
      case 2:
         return (X & Y) | (X & Z) | (Y & Z);
      case 3:
         return X ^ Y ^ Z;
      }
   return 0;
   }

// Circular shift function

libbase::int32u sha::cshift(const libbase::int32u x, const int s)
   {
   return (x << s) | (x >> (32-s));
   }

// Message expansion function

void sha::expand(const libbase::vector<libbase::int32u>& M, libbase::vector<libbase::int32u>& W)
   {
   // check input size
   assert(M.size() == 16);
   // set up output size
   W.init(80);
   // initialize values
   int i;
   for(i=0; i<16; i++)
      W(i) = M(i);
   for(i=16; i<80; i++)
      W(i) = cshift(W(i-3) ^ W(i-8) ^ W(i-14) ^ W(i-16), 1);
   }

}; // end namespace
