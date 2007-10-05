#include "sha.h"

#include <sstream>

namespace libcomm {

using libbase::int8u;
using libbase::int32u;
using libbase::vector;
using std::string;

const libbase::vcs sha::version("Secure Hash Algorithm module (sha)", 1.10);

//////////////////////////////////////////////////////////////////////
// Const values
//////////////////////////////////////////////////////////////////////

const int32u sha::K[] = { 0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xca62c1d6 };

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

sha::sha()
   {
   // reset size counter
   m_size = 0;
   // reset chaining variables
   m_hash.init(5);
   m_hash = 0;
   }

sha::~sha()
   {
   }

//////////////////////////////////////////////////////////////////////
// Conversion to/from strings
//////////////////////////////////////////////////////////////////////

sha::sha(const string& s)
   {
   // reset size counter
   m_size = 0;
   // reset chaining variables
   m_hash.init(5);
   // load from string
   std::istringstream is(s);
   is >> *this;
   }

sha::operator string() const
   {
   // write into a string
   std::ostringstream sout;
   sout << *this;
   return sout.str();
   }

//////////////////////////////////////////////////////////////////////
// Public interface for computing digest
//////////////////////////////////////////////////////////////////////

void sha::reset()
   {
   // reset size counter
   m_size = 0;
   // reset chaining variables
   m_hash.init(5);
   m_hash(0) = 0x67452301;
   m_hash(1) = 0xefcdab89;
   m_hash(2) = 0x98badcfe;
   m_hash(3) = 0x10325476;
   m_hash(4) = 0xc3d2e1f0;
#ifdef _DEBUG
   // debugging
   m_padded = m_terminated = false;
#endif
   }

void sha::process(const vector<int32u>& M)
   {
   // create expanded message block
   vector<int32u> W;
   expand(M, W);
   // copy variables
   vector<int32u> hash = m_hash;
   // main loop
   for(int t=0; t<80; t++)
      {
      const int32u temp = cshift(hash(0),5) + f(t,hash(1),hash(2),hash(3)) + hash(4) + W(t) + K[t/20];
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

void sha::process(const char *buf, const int size)
   {
   assert(size <= 64);
   //trace << "SHA: process block size " << size << "\n";
   // convert message block and process
   vector<int32u> M(16);
   // initialize values
   M = 0;
   for(int i=0; i<size; i++)
      M(i>>2) |= int8u(buf[i]) << 8*(3-(i & 3));
   // add padding (1-bit followed by zeros) if it fits and is necessary
   if(size < 64 && (m_size % 64) == 0)
      {
      M(size>>2) |= int8u(0x80) << 8*(3-(size & 3));
#ifdef _DEBUG
      if(m_padded)
         libbase::trace << "SHA Error: Padding already added\n";
      m_padded = true;
      //trace << "SHA: adding padding\n";
#endif
      }
   // update size counter
   m_size += size;
   // add file size (in bits) if this fits
   // (note that we need to fit the 8-byte size AND 1 byte of padding)
   if(size < 64-8)
      {
      M(14) = int32u(m_size >> 29);
      M(15) = int32u(m_size << 3);
#ifdef _DEBUG
      m_terminated = true;
      //trace << "SHA: adding file size " << hex << M(14) << " " << M(15) << dec << "\n";
#endif
      }
   // go through the SHA algorithm
   process(M);
   }

void sha::process(std::istream& sin)
   {
   char buf[64];
   // initialize the variables
   reset();
   // process whole data stream
   while(!sin.eof())
      {
      sin.read(buf, 64);
      process(buf, sin.gcount());
      }
   // if necessary, flush to include stream length
   if(sin.gcount() >= 64-8)
      process(buf, 0);
   }

//////////////////////////////////////////////////////////////////////
// Comparison functions
//////////////////////////////////////////////////////////////////////

bool sha::operator>(const sha& x) const
   {
   for(int i=0; i<5; i++)
      {
      if(m_hash(i) > x.m_hash(i))
         return true;
      if(m_hash(i) < x.m_hash(i))
         return false;
      }
   return false;
   }

bool sha::operator<(const sha& x) const
   {
   for(int i=0; i<5; i++)
      {
      if(m_hash(i) < x.m_hash(i))
         return true;
      if(m_hash(i) > x.m_hash(i))
         return false;
      }
   return false;
   }

bool sha::operator==(const sha& x) const
   {
   for(int i=0; i<5; i++)
      if(m_hash(i) != x.m_hash(i))
         return false;
   return true;
   }

bool sha::operator!=(const sha& x) const
   {
   return !operator==(x);
   }

//////////////////////////////////////////////////////////////////////
// SHA nonlinear function implementations
//////////////////////////////////////////////////////////////////////

int32u sha::f(const int t, const int32u X, const int32u Y, const int32u Z)
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

//////////////////////////////////////////////////////////////////////
// Circular shift function
//////////////////////////////////////////////////////////////////////

int32u sha::cshift(const int32u x, const int s)
   {
   return (x << s) | (x >> (32-s));   
   }

//////////////////////////////////////////////////////////////////////
// Message expansion function
//////////////////////////////////////////////////////////////////////

void sha::expand(const vector<int32u>& M, vector<int32u>& W)
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

//////////////////////////////////////////////////////////////////////
// Stream input/output
//////////////////////////////////////////////////////////////////////

std::ostream& operator<<(std::ostream& sout, const sha& x)
   {
#ifdef _DEBUG
   if(!x.m_padded)
      libbase::trace << "SHA Error: Unpadded stream\n";
   if(!x.m_terminated)
      libbase::trace << "SHA Error: Unterminated stream\n";
#endif
   const std::ios::fmtflags flags = sout.flags();
   for(int i=0; i<5; i++)
      {
      sout.width(8);
      sout.fill('0');
      sout << std::hex << x.m_hash(i);
      }
   sout.flags(flags);
   return sout;
   }

std::istream& operator>>(std::istream& sin, sha& x)
   {
   const std::ios::fmtflags flags = sin.flags();
   sin >> std::ws;
   char buf[9];
   buf[8] = 0;
   for(int i=0; i<5; i++)
      {
      sin.read(buf, 8);
      //trace << "Reading hash " << i << ": " << buf << "\n";
      x.m_hash(i) = strtoul(buf, NULL, 16);
      //trace << "Converted as " << hex << x.m_hash(i) << dec << "\n";
      }
   sin.flags(flags);
   return sin;
   }

}; // end namespace
