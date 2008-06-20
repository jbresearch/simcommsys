/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "md5.h"

#include <math.h>
#include <sstream>

namespace libcomm {

// Static values

bool md5::tested = false;
libbase::vector<libbase::int32u> md5::t;

// Const values

const int md5::s[] = { 7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22, \
                       5, 9, 14, 20,   5, 9, 14, 20,   5, 9, 14, 20,   5, 9, 14, 20,  \
                       4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23, \
                       6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21  };

const int md5::ndx[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, \
                         1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, \
                         5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, \
                         0, 7, 14, 5, 12, 3, 10, 1, 8, 15, 6, 13, 4, 11, 2, 9  };

// Construction/Destruction

md5::md5()
   {
   // reset size counter
   m_size = 0;
   // reset chaining variables
   m_hash.init(4);
   m_hash = 0;
   // initialise constants if not yet done
   if(t.size() == 0)
      {
      t.init(64);
      for(int i=0; i<64; i++)
         {
         t(i) = libbase::int32u(floor(pow(double(2),32) * fabs(sin(double(i+1)))));
         //trace << "t(" << i << ") = " << hex << t(i) << dec << "\n";
         }
      }
   // perform implementation tests on algorithm, exit on failure
   if(!tested)
      {
      libbase::trace << "md5: Testing implementation\n";
      // http://www.faqs.org/rfcs/rfc1321.html
      std::string sMessage, sHash;
      // Test libbase::vector 0
      sMessage = "";
      sHash = "d41d8cd98f00b204e9800998ecf8427e";
      assert(verify(sMessage,sHash));
      // Test libbase::vector 1
      sMessage = "a";
      sHash = "0cc175b9c0f1b6a831c399e269772661";
      assert(verify(sMessage,sHash));
      // Test libbase::vector 2
      sMessage = "abc";
      sHash = "900150983cd24fb0d6963f7d28e17f72";
      assert(verify(sMessage,sHash));
      // Test libbase::vector 3
      sMessage = "message digest";
      sHash = "f96b697d7cb7938d525a2f31aaf161d0";
      assert(verify(sMessage,sHash));
      // Test libbase::vector 4
      sMessage = "abcdefghijklmnopqrstuvwxyz";
      sHash = "c3fcd3d76192e4007dfb496cca67e13b";
      assert(verify(sMessage,sHash));
      // Test libbase::vector 5
      sMessage = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
      sHash = "d174ab98d277d9f5a5611c2c9f419d9f";
      assert(verify(sMessage,sHash));
      // Test libbase::vector 6
      sMessage = "12345678901234567890123456789012345678901234567890123456789012345678901234567890";
      sHash = "57edf4a22be3c955ac49da2e2107b67a";
      assert(verify(sMessage,sHash));
      // return ok
      tested = true;
      }
   }

// Conversion to/from strings

md5::md5(const std::string& s)
   {
   // reset size counter
   m_size = 0;
   // reset chaining variables
   m_hash.init(4);
   // load from std::string
   std::istringstream is(s);
   is >> *this;
   }

md5::operator std::string() const
   {
   // write into a std::string
   std::ostringstream sout;
   sout << *this;
   return sout.str();
   }

// Public interface for computing digest

void md5::reset()
   {
   // reset size counter
   m_size = 0;
   // reset chaining variables
   m_hash.init(4);
   m_hash(0) = 0x67452301;
   m_hash(1) = 0xefcdab89;
   m_hash(2) = 0x98badcfe;
   m_hash(3) = 0x10325476;
#ifndef NDEBUG
   // debugging
   m_padded = m_terminated = false;
#endif
   }

void md5::process(const libbase::vector<libbase::int32u>& M)
   {
   // copy variables
   libbase::vector<libbase::int32u> hash = m_hash;
   // main loop
   for(int i=0; i<64; i++)
      {
      int a = (64-i) & 0x3;
      int b = (65-i) & 0x3;
      int c = (66-i) & 0x3;
      int d = (67-i) & 0x3;
      hash(a) = op(i, hash(a), hash(b), hash(c), hash(d), M);
      }
   // add back variables
   m_hash += hash;
   }

void md5::process(const char *buf, const int size)
   {
   using libbase::trace;
   assert(size <= 64);
   //trace << "MD5: process block size " << size << "\n";
   // convert message block and process
   libbase::vector<libbase::int32u> M(16);
   // initialize values
   M = 0;
   trace << "md5: block input = " << std::hex;
   for(int i=0; i<size; i++)
      {
      M(i>>2) |= libbase::int8u(buf[i]) << 8*(i & 3);
#ifndef NDEBUG
      trace.width(2);
      trace.fill('0');
      trace << int(libbase::int8u(buf[i])) << " ";
#endif
      }
   trace << std::dec << "\n";
   // add padding (1-bit followed by zeros) if it fits and is necessary
   if(size < 64 && (m_size % 64) == 0)
      {
      M(size>>2) |= libbase::int8u(0x80) << 8*(size & 3);
#ifndef NDEBUG
      if(m_padded)
         trace << "MD5 Error: Padding already added\n";
      m_padded = true;
      //trace << "MD5: adding padding\n";
#endif
      }
   // update size counter
   m_size += size;
   // add file size (in bits) if this fits
   // (note that we need to fit the 8-byte size AND 1 byte of padding)
   if(size < 64-8)
      {
      M(14) = libbase::int32u(m_size << 3);
      M(15) = libbase::int32u(m_size >> 29);
#ifndef NDEBUG
      m_terminated = true;
      //trace << "MD5: adding file size " << hex << M(14) << " " << M(15) << dec << "\n";
#endif
      }
   // go through the MD5 algorithm
   process(M);
   }

void md5::process(std::istream& sin)
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
   // if necessary, flush to include padding and stream length
   if(sin.gcount() >= 64-8)
      process(buf, 0);
   }

// Comparison functions

bool md5::operator>(const md5& x) const
   {
   for(int i=0; i<4; i++)
      {
      if(m_hash(i) > x.m_hash(i))
         return true;
      if(m_hash(i) < x.m_hash(i))
         return false;
      }
   return false;
   }

bool md5::operator<(const md5& x) const
   {
   for(int i=0; i<4; i++)
      {
      if(m_hash(i) < x.m_hash(i))
         return true;
      if(m_hash(i) > x.m_hash(i))
         return false;
      }
   return false;
   }

bool md5::operator==(const md5& x) const
   {
   for(int i=0; i<4; i++)
      if(m_hash(i) != x.m_hash(i))
         return false;
   return true;
   }

bool md5::operator!=(const md5& x) const
   {
   return !operator==(x);
   }

// Verification function

bool md5::verify(const std::string message, const std::string hash)
   {
   reset();
   // process requires a pass by reference, which cannot be done by
   // direct conversion.
   std::istringstream s(message);
   process(s);
   return hash == std::string(*this);
   }

// Circular shift function

libbase::int32u md5::cshift(const libbase::int32u x, const int s)
   {
   return (x << s) | (x >> (32-s));
   }

// MD5 nonlinear function implementations

libbase::int32u md5::f(const int i, const libbase::int32u X, const libbase::int32u Y, const libbase::int32u Z)
   {
   assert(i<64);
   switch(i/16)
      {
      case 0:
         return (X & Y) | ((~X) & Z);
      case 1:
         return (X & Z) | (Y & (~Z));
      case 2:
         return X ^ Y ^ Z;
      case 3:
         return Y ^ (X | (~Z));
      }
   return 0;
   }

// Circular shift function

libbase::int32u md5::op(const int i, const libbase::int32u a, const libbase::int32u b, const libbase::int32u c, const libbase::int32u d, const libbase::vector<libbase::int32u>& M)
   {
   return b + cshift(a + f(i,b,c,d) + M(ndx[i]) + t(i), s[i]);
   }

// Stream input/output

std::ostream& operator<<(std::ostream& sout, const md5& x)
   {
#ifndef NDEBUG
   using libbase::trace;
   if(!x.m_padded)
      trace << "MD5 Error: Unpadded stream\n";
   if(!x.m_terminated)
      trace << "MD5 Error: Unterminated stream\n";
#endif
   const std::ios::fmtflags flags = sout.flags();
   sout << std::hex;
   for(int i=0; i<4; i++)
      for(int j=0; j<32; j+=8)
         {
         sout.width(2);
         sout.fill('0');
         sout << ((x.m_hash(i) >> j) & 0xff);
         }
   sout.flags(flags);
   return sout;
   }

std::istream& operator>>(std::istream& sin, md5& x)
   {
   const std::ios::fmtflags flags = sin.flags();
   sin >> std::ws;
   char buf[9];
   buf[8] = 0;
   for(int i=0; i<4; i++)
      {
      sin.read(buf+6, 2);
      sin.read(buf+4, 2);
      sin.read(buf+2, 2);
      sin.read(buf+0, 2);
      x.m_hash(i) = strtoul(buf, NULL, 16);
      }
   sin.flags(flags);
   return sin;
   }

}; // end namespace
