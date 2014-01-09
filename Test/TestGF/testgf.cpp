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

#include "gf.h"
#include "gf_fast.h"
#include "bitfield.h"
#include "matrix.h"
#include "cputimer.h"
#include <iostream>

namespace testgf {

using libbase::gf;
using libbase::gf_fast;
using libbase::bitfield;
using libbase::matrix;
using libbase::cputimer;

using std::cout;
using std::cerr;
using std::hex;
using std::dec;

/*!
 * \brief Exponential table entries for base {03}
 * cf. Gladman, "A Specification for Rijndael, the AES Algorithm", 2003, p.5
 */
const int aestable[] = {0x01, 0x03, 0x05, 0x0f, 0x11, 0x33, 0x55, 0xff, 0x1a,
      0x2e, 0x72, 0x96, 0xa1, 0xf8, 0x13, 0x35, 0x5f, 0xe1, 0x38, 0x48, 0xd8,
      0x73, 0x95, 0xa4, 0xf7, 0x02, 0x06, 0x0a, 0x1e, 0x22, 0x66, 0xaa, 0xe5,
      0x34, 0x5c, 0xe4, 0x37, 0x59, 0xeb, 0x26, 0x6a, 0xbe, 0xd9, 0x70, 0x90,
      0xab, 0xe6, 0x31, 0x53, 0xf5, 0x04, 0x0c, 0x14, 0x3c, 0x44, 0xcc, 0x4f,
      0xd1, 0x68, 0xb8, 0xd3, 0x6e, 0xb2, 0xcd, 0x4c, 0xd4, 0x67, 0xa9, 0xe0,
      0x3b, 0x4d, 0xd7, 0x62, 0xa6, 0xf1, 0x08, 0x18, 0x28, 0x78, 0x88, 0x83,
      0x9e, 0xb9, 0xd0, 0x6b, 0xbd, 0xdc, 0x7f, 0x81, 0x98, 0xb3, 0xce, 0x49,
      0xdb, 0x76, 0x9a, 0xb5, 0xc4, 0x57, 0xf9, 0x10, 0x30, 0x50, 0xf0, 0x0b,
      0x1d, 0x27, 0x69, 0xbb, 0xd6, 0x61, 0xa3, 0xfe, 0x19, 0x2b, 0x7d, 0x87,
      0x92, 0xad, 0xec, 0x2f, 0x71, 0x93, 0xae, 0xe9, 0x20, 0x60, 0xa0, 0xfb,
      0x16, 0x3a, 0x4e, 0xd2, 0x6d, 0xb7, 0xc2, 0x5d, 0xe7, 0x32, 0x56, 0xfa,
      0x15, 0x3f, 0x41, 0xc3, 0x5e, 0xe2, 0x3d, 0x47, 0xc9, 0x40, 0xc0, 0x5b,
      0xed, 0x2c, 0x74, 0x9c, 0xbf, 0xda, 0x75, 0x9f, 0xba, 0xd5, 0x64, 0xac,
      0xef, 0x2a, 0x7e, 0x82, 0x9d, 0xbc, 0xdf, 0x7a, 0x8e, 0x89, 0x80, 0x9b,
      0xb6, 0xc1, 0x58, 0xe8, 0x23, 0x65, 0xaf, 0xea, 0x25, 0x6f, 0xb1, 0xc8,
      0x43, 0xc5, 0x54, 0xfc, 0x1f, 0x21, 0x63, 0xa5, 0xf4, 0x07, 0x09, 0x1b,
      0x2d, 0x77, 0x99, 0xb0, 0xcb, 0x46, 0xca, 0x45, 0xcf, 0x4a, 0xde, 0x79,
      0x8b, 0x86, 0x91, 0xa8, 0xe3, 0x3e, 0x42, 0xc6, 0x51, 0xf3, 0x0e, 0x12,
      0x36, 0x5a, 0xee, 0x29, 0x7b, 0x8d, 0x8c, 0x8f, 0x8a, 0x85, 0x94, 0xa7,
      0xf2, 0x0d, 0x17, 0x39, 0x4b, 0xdd, 0x7c, 0x84, 0x97, 0xa2, 0xfd, 0x1c,
      0x24, 0x6c, 0xb4, 0xc7, 0x52, 0xf6, 0x01};

void TestBinaryField()
   {
   // Create values in the Binary field GF(2): m(x) = 1 { 1 }
   typedef gf<1, 0x3> Binary;
   // Compute and display addition & multiplication tables
   cout << std::endl << "Binary Addition table:" << std::endl;
   for (int x = 0; x < 2; x++)
      for (int y = 0; y < 2; y++)
         cout << Binary(x) + Binary(y) << (y == 1 ? '\n' : '\t');
   cout << std::endl << "Binary Multiplication table:" << std::endl;
   for (int x = 0; x < 2; x++)
      for (int y = 0; y < 2; y++)
         cout << Binary(x) * Binary(y) << (y == 1 ? '\n' : '\t');
   }

void TestRijndaelField()
   {
   // Create a value in the Rijndael field GF(2^8): m(x) = 1 { 0001 1011 }
   gf<8, 0x11B> E = 1;
   // Compute and display exponential table using {03} as a multiplier
   // using the tabular format in Gladman.
   cout << std::endl << "Rijndael GF(2^8) exponentiation table:" << std::endl;
   cout << hex;
   for (int x = 0; x < 16; x++)
      for (int y = 0; y < 16; y++)
         {
         assert(E == aestable[(x << 4) + y]);
         cout << int(E) << (y == 15 ? '\n' : '\t');
         E *= 3;
         }
   cout << dec;
   }

template <int m, int poly>
void ListField()
   {
   // Compute and display exponential table using {2} as a multiplier
   cout << std::endl << "GF(" << m << ",0x" << hex << poly << dec << ") table:" << std::endl;
   cout << 0 << '\t' << 0 << '\t' << bitfield(0, m) << std::endl;
   gf<m, poly> E = 1;
   for (int x = 1; x < gf<m, poly>::elements(); x++)
      {
      cout << x << "\ta" << x - 1 << '\t' << bitfield(E, m) << std::endl;
      E *= 2;
      }
   }

template <int m, int poly>
void TestMulDiv()
   {
   // Compute and display exponential table using {2} as a multiplier
   cout << std::endl << "GF(" << m << ",0x" << hex << poly << dec
         << ") multiplication/division:" << std::endl;
   cout << "power\tvalue\tinverse\tmul" << std::endl;
   gf<m, poly> E = 1;
   for (int x = 1; x < gf<m, poly>::elements(); x++)
      {
      cout << "a" << x - 1 << '\t' << bitfield(E, m) << '\t' << bitfield(
            E.inverse(), m) << '\t' << bitfield(E.inverse() * E, m) << std::endl;
      E *= 2;
      }
   }

void TestGenPowerGF2()
   {
   cout << std::endl << "Binary generator matrix power sequence:" << std::endl;
   // Create values in the Binary field GF(2): m(x) = 1 { 1 }
   typedef gf<1, 0x3> Binary;
   // Create generator matrix for DVB-CRSC code:
   matrix<Binary> G(3, 3);
   G = 0;
   G(0, 0) = 1;
   G(2, 0) = 1;
   G(0, 1) = 1;
   G(1, 2) = 1;
   // Compute and display first 8 powers of G
   for (int i = 0; i < 8; i++)
      {
      cout << "G^" << i << " = " << std::endl;
      pow(G, i).serialize(cout);
      }
   }

void TestGenPowerGF8()
   {
   cout << std::endl << "GF(8) generator matrix power sequence:" << std::endl;
   // Create values in the field GF(8): m(x) = 1 { 011 }
   typedef gf<3, 0xB> GF8;
   // Create generator matrix:
   matrix<GF8> G(2, 2);
   G(0, 0) = 1;
   G(1, 0) = 6;
   G(0, 1) = 1;
   G(1, 1) = 0;
   // Compute and display first 16 powers of G
   for (int i = 0; i < 16; i++)
      {
      cout << "G^" << i << " = " << std::endl;
      pow(G, i).serialize(cout);
      }
   }

void TestFastGF2()
   {
   gf<1, 0x3> gf1, gf2, gf3;
   gf_fast<1, 0x3> gffast1, gffast2, gffast3;
   //initialise the old gf
   gf1 = gf<1, 0x3> (0);
   gf2 = gf<1, 0x3> (1);

   cout << std::endl << "Checking gf_fast<1.0x3> against gf<1,0x3>" << std::endl;
   //init the new gf
   gffast1 = gf_fast<1, 0x3> (0);
   gffast2 = gf_fast<1, 0x3> (1);

   gf3 = gf1 * gf2;
   gffast3 = gffast1 * gffast2;
   assert(gffast3 == gf3);
   assert(gffast3.log_gf() == 0);//by convention

   gf3 = gf2 * gf2;
   gffast3 = gffast2 * gffast2;
   assert(gffast3 == gf3);
   assert(gffast3.log_gf() == 0);

   gf3 = gf1 + gf2;
   gffast3 = gffast1 + gffast2;
   assert(gffast3 == gf3);
   assert(gffast3.log_gf() == 0);

   gf3 = gf1 / gf2;
   gffast3 = gffast1 / gffast2;
   assert(gffast3 == gf3);
   assert(gffast3.log_gf() == 0);//by convention

   }

void TestFastGF4()
   {
   gf<2, 0x7> gf1, gf2, gf3, gf4;
   gf_fast<2, 0x7> gffast1, gffast2, gffast3, gffast4;
   int non_zero = 3;
   int num_of_elements = 4;
   int power = 0;

   cout << std::endl << "Checking gf_fast<2, 0x7> against gf<2, 0x7>" << std::endl;
   cout << "Checking the powers" << std::endl;
   //check the powers
   gf1 = gf<2, 0x7> (2);//alpha
   gf3 = gf1;
   gffast1 = gf_fast<2, 0x7> (2);//alpgha
   gffast2 = gffast1;
   gffast3 = gffast1;

   for (int loop1 = -20; loop1 < 20; loop1++)
      {
      gf3 = gf3 * gf1;
      gffast3 = gffast3 * gffast1;
      gffast4 = gffast2 .power(loop1);
      assert(gf3 == gffast3);
      if (loop1 < 0)
         {
         power = (loop1 % non_zero);
         if (power < 0)
            power += non_zero;
         gf2 = gf1.inverse();
         gf4 = gf<2, 0x7> (1);
         for (int loop2 = loop1; loop2 < 0; loop2++)
            gf4 = gf4 * gf2;
         assert((gffast4 == gf4) && gffast4.log_gf() == power);
         }
      else if (loop1 == 0)
         {
         assert((1 == gffast4) && (gffast4.log_gf() == 0));
         }
      else
         {
         power = (loop1 % non_zero);
         gf4 = gf<2, 0x7> (1);
         gf2 = gf1;
         for (int loop2 = 0; loop2 < loop1; loop2++)
            gf4 = gf4 * gf2;
         assert((gffast4 == gf4) && gffast4.log_gf() == power);
         }
      }

   cout << "Checking addition, subtraction,multiplication and division" << std::endl;
   //check addition, subtraction, multiplication and division
   int tmp_val;
   for (int loop1 = 0; loop1 < num_of_elements; loop1++)
      {
      for (int loop2 = 0; loop2 < num_of_elements; loop2++)
         {
         gf1 = gf<2, 0x7> (loop1);
         gf2 = gf<2, 0x7> (loop2);

         gffast1 = gf_fast<2, 0x7> (loop1);
         gffast2 = gf_fast<2, 0x7> (loop2);

         //addition
         gf3 = gf1 + gf2;
         gffast3 = gffast1 + gffast2;
         assert(gf3 == gffast3);
         tmp_val = gf3;
         gffast4 = gf_fast<2, 0x7> (tmp_val);
         assert(gffast4.log_gf() == gffast3.log_gf());

         //subtraction
         gf3 = gf1 - gf2;
         gffast3 = gffast1 - gffast2;
         assert(gf3 == gffast3);
         tmp_val = gf3;
         gffast4 = gf_fast<2, 0x7> (tmp_val);
         assert(gffast4.log_gf() == gffast3.log_gf());

         //multiplication
         gf3 = gf1 * gf2;
         gffast3 = gffast1 * gffast2;
         assert(gf3 == gffast3);
         tmp_val = gf3;
         gffast4 = gf_fast<2, 0x7> (tmp_val);
         assert(gffast4.log_gf() == gffast3.log_gf());

         //division
         if (loop2 != 0) //don't divide by 0!
            {
            gf3 = gf1 / gf2;
            gffast3 = gffast1 / gffast2;
            assert(gf3 == gffast3);
            tmp_val = gf3;
            gffast4 = gf_fast<2, 0x7> (tmp_val);
            assert(gffast4.log_gf() == gffast3.log_gf());
            }
         }
      }

   cout << "Checking inverses" << std::endl;
   //check the inverses
   for (int loop1 = 1; loop1 < num_of_elements; loop1++)
      {
      gf1 = gf<2, 0x7> (loop1);
      gffast1 = gf_fast<2, 0x7> (loop1);
      gf2 = gf1.inverse();
      gffast2 = gffast1.inverse();
      tmp_val = gf2;
      gffast3 = gf_fast<2, 0x7> (tmp_val);
      assert((gf2 == gffast2) && (gffast2.log_gf() == gffast3.log_gf()));
      }

   cputimer t1;
   cout << "Proving that gf_fast is faster at multiplication:" << std::endl;
   cout
         << "multiplying all non-zero field elements together 1,000,000 times in gf<2,0x7>" << std::endl;
   t1.start();
   gf1 = gf<2, 0x7> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gf1 = gf1 * gf<2, 0x7> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;

   cout
         << "multiplying all non-zero field elements together 1,000,000 times in gf_fast<2, 0x7>" << std::endl;
   t1.start();
   gffast1 = gf_fast<2, 0x7> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gffast1 = gffast1 * gf_fast<2, 0x7> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;

   cout << "Proving that gf_fast is as fast at additions as gf:" << std::endl;
   cout
         << "adding all non-zero field elements together 1,000,000 times in gf" << std::endl;
   t1.start();
   gf1 = gf<2, 0x7> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gf1 = gf1 + gf<2, 0x7> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;

   cout
         << "adding all non-zero field elements together 1,000,000 times in gf_fast" << std::endl;
   t1.start();
   gffast1 = gf_fast<2, 0x7> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gffast1 = gffast1 + gf_fast<2, 0x7> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;
   }

void TestFastGF8()
   {
   gf<3, 0xB> gf1, gf2, gf3, gf4;
   gf_fast<3, 0xB> gffast1, gffast2, gffast3, gffast4;
   int non_zero = 7;
   int num_of_elements = 8;
   int power = 0;

   cout << std::endl << "Checking gf_fast<3, 0xB> against gf<3, 0xB>" << std::endl;
   cout << "Checking the powers" << std::endl;
   //check the powers
   gf1 = gf<3, 0xB> (2);//alpha
   gf3 = gf1;
   gffast1 = gf_fast<3, 0xB> (2);//alpgha
   gffast2 = gffast1;
   gffast3 = gffast1;

   for (int loop1 = -20; loop1 < 20; loop1++)
      {
      gf3 = gf3 * gf1;
      gffast3 = gffast3 * gffast1;
      gffast4 = gffast2 .power(loop1);
      assert(gf3 == gffast3);
      if (loop1 < 0)
         {
         power = (loop1 % non_zero);
         if (power < 0)
            power += non_zero;
         gf2 = gf1.inverse();
         gf4 = gf<3, 0xB> (1);
         for (int loop2 = loop1; loop2 < 0; loop2++)
            gf4 = gf4 * gf2;
         assert((gffast4 == gf4) && gffast4.log_gf() == power);
         }
      else if (loop1 == 0)
         {
         assert((1 == gffast4) && (gffast4.log_gf() == 0));
         }
      else
         {
         power = (loop1 % non_zero);
         gf4 = gf<3, 0xB> (1);
         gf2 = gf1;
         for (int loop2 = 0; loop2 < loop1; loop2++)
            gf4 = gf4 * gf2;
         assert((gffast4 == gf4) && gffast4.log_gf() == power);
         }
      }

   cout << "Checking addition, subtraction,multiplication and division" << std::endl;
   //check addition, subtraction, multiplication and division
   int tmp_val;
   for (int loop1 = 0; loop1 < num_of_elements; loop1++)
      {
      for (int loop2 = 0; loop2 < num_of_elements; loop2++)
         {
         gf1 = gf<3, 0xB> (loop1);
         gf2 = gf<3, 0xB> (loop2);

         gffast1 = gf_fast<3, 0xB> (loop1);
         gffast2 = gf_fast<3, 0xB> (loop2);

         //addition
         gf3 = gf1 + gf2;
         gffast3 = gffast1 + gffast2;
         assert(gf3 == gffast3);
         tmp_val = gf3;
         gffast4 = gf_fast<3, 0xB> (tmp_val);
         assert(gffast4.log_gf() == gffast3.log_gf());

         //subtraction
         gf3 = gf1 - gf2;
         gffast3 = gffast1 - gffast2;
         assert(gf3 == gffast3);
         tmp_val = gf3;
         gffast4 = gf_fast<3, 0xB> (tmp_val);
         assert(gffast4.log_gf() == gffast3.log_gf());

         //multiplication
         gf3 = gf1 * gf2;
         gffast3 = gffast1 * gffast2;
         assert(gf3 == gffast3);
         tmp_val = gf3;
         gffast4 = gf_fast<3, 0xB> (tmp_val);
         assert(gffast4.log_gf() == gffast3.log_gf());

         //division
         if (loop2 != 0) //don't divide by 0!
            {
            gf3 = gf1 / gf2;
            gffast3 = gffast1 / gffast2;
            assert(gf3 == gffast3);
            tmp_val = gf3;
            gffast4 = gf_fast<3, 0xB> (tmp_val);
            assert(gffast4.log_gf() == gffast3.log_gf());
            }
         }
      }

   cout << "Checking inverses" << std::endl;
   //check the inverses
   for (int loop1 = 1; loop1 < num_of_elements; loop1++)
      {
      gf1 = gf<3, 0xB> (loop1);
      gffast1 = gf_fast<3, 0xB> (loop1);
      gf2 = gf1.inverse();
      gffast2 = gffast1.inverse();
      tmp_val = gf2;
      gffast3 = gf_fast<3, 0xB> (tmp_val);
      assert((gf2 == gffast2) && (gffast2.log_gf() == gffast3.log_gf()));
      }

   cputimer t1;
   cout << "Proving that gf_fast is faster at multiplication:" << std::endl;
   cout
         << "multiplying all non-zero field elements together 1,000,000 times in gf<3, 0xB>" << std::endl;
   t1.start();
   gf1 = gf<3, 0xB> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gf1 = gf1 * gf<3, 0xB> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;

   cout
         << "multiplying all non-zero field elements together 1,000,000 times in gf_fast<3, 0xB>" << std::endl;
   t1.start();
   gffast1 = gf_fast<3, 0xB> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gffast1 = gffast1 * gf_fast<3, 0xB> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;

   cout << "Proving that gf_fast is as fast at additions as gf:" << std::endl;
   cout
         << "adding all non-zero field elements together 1,000,000 times in gf" << std::endl;
   t1.start();
   gf1 = gf<3, 0xB> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gf1 = gf1 + gf<3, 0xB> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;

   cout
         << "adding all non-zero field elements together 1,000,000 times in gf_fast" << std::endl;
   t1.start();
   gffast1 = gf_fast<3, 0xB> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gffast1 = gffast1 + gf_fast<3, 0xB> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;

   }

void TestFastGF16()
   {
   gf<4, 0x13> gf1, gf2, gf3, gf4;
   gf_fast<4, 0x13> gffast1, gffast2, gffast3, gffast4;
   int non_zero = 15;
   int num_of_elements = 16;
   int power = 0;

   cout << std::endl << "Checking gf_fast<4, 0x13> against gf<4, 0x13>" << std::endl;
   cout << "Checking the powers" << std::endl;
   //check the powers
   gf1 = gf<4, 0x13> (2);//alpha
   gf3 = gf1;
   gffast1 = gf_fast<4, 0x13> (2);//alpgha
   gffast2 = gffast1;
   gffast3 = gffast1;

   for (int loop1 = -20; loop1 < 20; loop1++)
      {
      gf3 = gf3 * gf1;
      gffast3 = gffast3 * gffast1;
      gffast4 = gffast2 .power(loop1);
      assert(gf3 == gffast3);
      if (loop1 < 0)
         {
         power = (loop1 % non_zero);
         if (power < 0)
            power += non_zero;
         gf2 = gf1.inverse();
         gf4 = gf<4, 0x13> (1);
         for (int loop2 = loop1; loop2 < 0; loop2++)
            gf4 = gf4 * gf2;
         assert((gffast4 == gf4) && gffast4.log_gf() == power);
         }
      else if (loop1 == 0)
         {
         assert((1 == gffast4) && (gffast4.log_gf() == 0));
         }
      else
         {
         power = (loop1 % non_zero);
         gf4 = gf<4, 0x13> (1);
         gf2 = gf1;
         for (int loop2 = 0; loop2 < loop1; loop2++)
            gf4 = gf4 * gf2;
         assert((gffast4 == gf4) && gffast4.log_gf() == power);
         }
      }

   cout << "Checking addition, subtraction,multiplication and division" << std::endl;
   //check addition, subtraction, multiplication and division
   int tmp_val;
   for (int loop1 = 0; loop1 < num_of_elements; loop1++)
      {
      for (int loop2 = 0; loop2 < num_of_elements; loop2++)
         {
         gf1 = gf<4, 0x13> (loop1);
         gf2 = gf<4, 0x13> (loop2);

         gffast1 = gf_fast<4, 0x13> (loop1);
         gffast2 = gf_fast<4, 0x13> (loop2);

         //addition
         gf3 = gf1 + gf2;
         gffast3 = gffast1 + gffast2;
         assert(gf3 == gffast3);
         tmp_val = gf3;
         gffast4 = gf_fast<4, 0x13> (tmp_val);
         assert(gffast4.log_gf() == gffast3.log_gf());

         //subtraction
         gf3 = gf1 - gf2;
         gffast3 = gffast1 - gffast2;
         assert(gf3 == gffast3);
         tmp_val = gf3;
         gffast4 = gf_fast<4, 0x13> (tmp_val);
         assert(gffast4.log_gf() == gffast3.log_gf());

         //multiplication
         gf3 = gf1 * gf2;
         gffast3 = gffast1 * gffast2;
         assert(gf3 == gffast3);
         tmp_val = gf3;
         gffast4 = gf_fast<4, 0x13> (tmp_val);
         assert(gffast4.log_gf() == gffast3.log_gf());

         //division
         if (loop2 != 0) //don't divide by 0!
            {
            gf3 = gf1 / gf2;
            gffast3 = gffast1 / gffast2;
            assert(gf3 == gffast3);
            tmp_val = gf3;
            gffast4 = gf_fast<4, 0x13> (tmp_val);
            assert(gffast4.log_gf() == gffast3.log_gf());
            }
         }
      }

   cout << "Checking inverses" << std::endl;
   //check the inverses
   for (int loop1 = 1; loop1 < num_of_elements; loop1++)
      {
      gf1 = gf<4, 0x13> (loop1);
      gffast1 = gf_fast<4, 0x13> (loop1);
      gf2 = gf1.inverse();
      gffast2 = gffast1.inverse();
      tmp_val = gf2;
      gffast3 = gf_fast<4, 0x13> (tmp_val);
      assert((gf2 == gffast2) && (gffast2.log_gf() == gffast3.log_gf()));
      }

   cputimer t1;
   cout << "Proving that gf_fast is faster at multiplication:" << std::endl;
   cout
         << "multiplying all non-zero field elements together 1,000,000 times in gf<4, 0x13>" << std::endl;
   t1.start();
   gf1 = gf<4, 0x13> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gf1 = gf1 * gf<4, 0x13> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;

   cout
         << "multiplying all non-zero field elements together 1,000,000 times in gf_fast<4, 0x13>" << std::endl;
   t1.start();
   gffast1 = gf_fast<4, 0x13> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gffast1 = gffast1 * gf_fast<4, 0x13> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;

   cout << "Proving that gf_fast is as fast at additions as gf:" << std::endl;
   cout
         << "adding all non-zero field elements together 1,000,000 times in gf" << std::endl;
   t1.start();
   gf1 = gf<4, 0x13> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gf1 = gf1 + gf<4, 0x13> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;

   cout
         << "adding all non-zero field elements together 1,000,000 times in gf_fast" << std::endl;
   t1.start();
   gffast1 = gf_fast<4, 0x13> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gffast1 = gffast1 + gf_fast<4, 0x13> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;

   }

void TestFastGF32()
   {
   gf<5, 0x25> gf1, gf2, gf3, gf4;
   gf_fast<5, 0x25> gffast1, gffast2, gffast3, gffast4;
   int non_zero = 31;
   int num_of_elements = 32;
   int power = 0;

   cout << std::endl << "Checking gf_fast<5, 0x25> against gf<5, 0x25>" << std::endl;
   cout << "Checking the powers" << std::endl;
   //check the powers
   gf1 = gf<5, 0x25> (2);//alpha
   gf3 = gf1;
   gffast1 = gf_fast<5, 0x25> (2);//alpgha
   gffast2 = gffast1;
   gffast3 = gffast1;

   for (int loop1 = -20; loop1 < 20; loop1++)
      {
      gf3 = gf3 * gf1;
      gffast3 = gffast3 * gffast1;
      gffast4 = gffast2 .power(loop1);
      assert(gf3 == gffast3);
      if (loop1 < 0)
         {
         power = (loop1 % non_zero);
         if (power < 0)
            power += non_zero;
         gf2 = gf1.inverse();
         gf4 = gf<5, 0x25> (1);
         for (int loop2 = loop1; loop2 < 0; loop2++)
            gf4 = gf4 * gf2;
         assert((gffast4 == gf4) && gffast4.log_gf() == power);
         }
      else if (loop1 == 0)
         {
         assert((1 == gffast4) && (gffast4.log_gf() == 0));
         }
      else
         {
         power = (loop1 % non_zero);
         gf4 = gf<5, 0x25> (1);
         gf2 = gf1;
         for (int loop2 = 0; loop2 < loop1; loop2++)
            gf4 = gf4 * gf2;
         assert((gffast4 == gf4) && gffast4.log_gf() == power);
         }
      }

   cout << "Checking addition, subtraction,multiplication and division" << std::endl;
   //check addition, subtraction, multiplication and division
   int tmp_val;
   for (int loop1 = 0; loop1 < num_of_elements; loop1++)
      {
      for (int loop2 = 0; loop2 < num_of_elements; loop2++)
         {
         gf1 = gf<5, 0x25> (loop1);
         gf2 = gf<5, 0x25> (loop2);

         gffast1 = gf_fast<5, 0x25> (loop1);
         gffast2 = gf_fast<5, 0x25> (loop2);

         //addition
         gf3 = gf1 + gf2;
         gffast3 = gffast1 + gffast2;
         assert(gf3 == gffast3);
         tmp_val = gf3;
         gffast4 = gf_fast<5, 0x25> (tmp_val);
         assert(gffast4.log_gf() == gffast3.log_gf());

         //subtraction
         gf3 = gf1 - gf2;
         gffast3 = gffast1 - gffast2;
         assert(gf3 == gffast3);
         tmp_val = gf3;
         gffast4 = gf_fast<5, 0x25> (tmp_val);
         assert(gffast4.log_gf() == gffast3.log_gf());

         //multiplication
         gf3 = gf1 * gf2;
         gffast3 = gffast1 * gffast2;
         assert(gf3 == gffast3);
         tmp_val = gf3;
         gffast4 = gf_fast<5, 0x25> (tmp_val);
         assert(gffast4.log_gf() == gffast3.log_gf());

         //division
         if (loop2 != 0) //don't divide by 0!
            {
            gf3 = gf1 / gf2;
            gffast3 = gffast1 / gffast2;
            assert(gf3 == gffast3);
            tmp_val = gf3;
            gffast4 = gf_fast<5, 0x25> (tmp_val);
            assert(gffast4.log_gf() == gffast3.log_gf());
            }
         }
      }

   cout << "Checking inverses" << std::endl;
   //check the inverses
   for (int loop1 = 1; loop1 < num_of_elements; loop1++)
      {
      gf1 = gf<5, 0x25> (loop1);
      gffast1 = gf_fast<5, 0x25> (loop1);
      gf2 = gf1.inverse();
      gffast2 = gffast1.inverse();
      tmp_val = gf2;
      gffast3 = gf_fast<5, 0x25> (tmp_val);
      assert((gf2 == gffast2) && (gffast2.log_gf() == gffast3.log_gf()));
      }

   cputimer t1;
   cout << "Proving that gf_fast is faster at multiplication:" << std::endl;
   cout
         << "multiplying all non-zero field elements together 1,000,000 times in gf<5, 0x25>" << std::endl;
   t1.start();
   gf1 = gf<5, 0x25> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gf1 = gf1 * gf<5, 0x25> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;

   cout
         << "multiplying all non-zero field elements together 1,000,000 times in gf_fast<5, 0x25>" << std::endl;
   t1.start();
   gffast1 = gf_fast<5, 0x25> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gffast1 = gffast1 * gf_fast<5, 0x25> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;

   cout << "Proving that gf_fast is as fast at additions as gf:" << std::endl;
   cout
         << "adding all non-zero field elements together 1,000,000 times in gf" << std::endl;
   t1.start();
   gf1 = gf<5, 0x25> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gf1 = gf1 + gf<5, 0x25> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;

   cout
         << "adding all non-zero field elements together 1,000,000 times in gf_fast" << std::endl;
   t1.start();
   gffast1 = gf_fast<5, 0x25> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gffast1 = gffast1 + gf_fast<5, 0x25> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;
   }

void TestFastGF64()
   {
   gf<6, 0x43> gf1, gf2, gf3, gf4;
   gf_fast<6, 0x43> gffast1, gffast2, gffast3, gffast4;
   int non_zero = 63;
   int num_of_elements = 64;
   int power = 0;

   cout << std::endl << "Checking gf_fast<6, 0x43> against gf<6, 0x43>" << std::endl;
   cout << "Checking the powers" << std::endl;
   //check the powers
   gf1 = gf<6, 0x43> (2);//alpha
   gf3 = gf1;
   gffast1 = gf_fast<6, 0x43> (2);//alpgha
   gffast2 = gffast1;
   gffast3 = gffast1;

   for (int loop1 = -20; loop1 < 20; loop1++)
      {
      gf3 = gf3 * gf1;
      gffast3 = gffast3 * gffast1;
      gffast4 = gffast2 .power(loop1);
      assert(gf3 == gffast3);
      if (loop1 < 0)
         {
         power = (loop1 % non_zero);
         if (power < 0)
            power += non_zero;
         gf2 = gf1.inverse();
         gf4 = gf<6, 0x43> (1);
         for (int loop2 = loop1; loop2 < 0; loop2++)
            gf4 = gf4 * gf2;
         assert((gffast4 == gf4) && gffast4.log_gf() == power);
         }
      else if (loop1 == 0)
         {
         assert((1 == gffast4) && (gffast4.log_gf() == 0));
         }
      else
         {
         power = (loop1 % non_zero);
         gf4 = gf<6, 0x43> (1);
         gf2 = gf1;
         for (int loop2 = 0; loop2 < loop1; loop2++)
            gf4 = gf4 * gf2;
         assert((gffast4 == gf4) && gffast4.log_gf() == power);
         }
      }

   cout << "Checking addition, subtraction,multiplication and division" << std::endl;
   //check addition, subtraction, multiplication and division
   int tmp_val;
   for (int loop1 = 0; loop1 < num_of_elements; loop1++)
      {
      for (int loop2 = 0; loop2 < num_of_elements; loop2++)
         {
         gf1 = gf<6, 0x43> (loop1);
         gf2 = gf<6, 0x43> (loop2);

         gffast1 = gf_fast<6, 0x43> (loop1);
         gffast2 = gf_fast<6, 0x43> (loop2);

         //addition
         gf3 = gf1 + gf2;
         gffast3 = gffast1 + gffast2;
         assert(gf3 == gffast3);
         tmp_val = gf3;
         gffast4 = gf_fast<6, 0x43> (tmp_val);
         assert(gffast4.log_gf() == gffast3.log_gf());

         //subtraction
         gf3 = gf1 - gf2;
         gffast3 = gffast1 - gffast2;
         assert(gf3 == gffast3);
         tmp_val = gf3;
         gffast4 = gf_fast<6, 0x43> (tmp_val);
         assert(gffast4.log_gf() == gffast3.log_gf());

         //multiplication
         gf3 = gf1 * gf2;
         gffast3 = gffast1 * gffast2;
         assert(gf3 == gffast3);
         tmp_val = gf3;
         gffast4 = gf_fast<6, 0x43> (tmp_val);
         assert(gffast4.log_gf() == gffast3.log_gf());

         //division
         if (loop2 != 0) //don't divide by 0!
            {
            gf3 = gf1 / gf2;
            gffast3 = gffast1 / gffast2;
            assert(gf3 == gffast3);
            tmp_val = gf3;
            gffast4 = gf_fast<6, 0x43> (tmp_val);
            assert(gffast4.log_gf() == gffast3.log_gf());
            }
         }
      }

   cout << "Checking inverses" << std::endl;
   //check the inverses
   for (int loop1 = 1; loop1 < num_of_elements; loop1++)
      {
      gf1 = gf<6, 0x43> (loop1);
      gffast1 = gf_fast<6, 0x43> (loop1);
      gf2 = gf1.inverse();
      gffast2 = gffast1.inverse();
      tmp_val = gf2;
      gffast3 = gf_fast<6, 0x43> (tmp_val);
      assert((gf2 == gffast2) && (gffast2.log_gf() == gffast3.log_gf()));
      }

   cputimer t1;
   cout << "Proving that gf_fast is faster at multiplication:" << std::endl;
   cout
         << "multiplying all non-zero field elements together 1,000,000 times in gf<6, 0x43>" << std::endl;
   t1.start();
   gf1 = gf<6, 0x43> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gf1 = gf1 * gf<6, 0x43> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;

   cout
         << "multiplying all non-zero field elements together 1,000,000 times in gf_fast<6, 0x43>" << std::endl;
   t1.start();
   gffast1 = gf_fast<6, 0x43> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gffast1 = gffast1 * gf_fast<6, 0x43> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;

   cout << "Proving that gf_fast is as fast at additions as gf:" << std::endl;
   cout
         << "adding all non-zero field elements together 1,000,000 times in gf" << std::endl;
   t1.start();
   gf1 = gf<6, 0x43> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gf1 = gf1 + gf<6, 0x43> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;

   cout
         << "adding all non-zero field elements together 1,000,000 times in gf_fast" << std::endl;
   t1.start();
   gffast1 = gf_fast<6, 0x43> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gffast1 = gffast1 + gf_fast<6, 0x43> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;
   }

void TestFastGF128()
   {
   gf<7, 0x89> gf1, gf2, gf3, gf4;
   gf_fast<7, 0x89> gffast1, gffast2, gffast3, gffast4;
   int non_zero = 127;
   int num_of_elements = 128;
   int power = 0;

   cout << std::endl << "Checking gf_fast<7, 0x89> against gf<7, 0x89>" << std::endl;
   cout << "Checking the powers" << std::endl;
   //check the powers
   gf1 = gf<7, 0x89> (2);//alpha
   gf3 = gf1;
   gffast1 = gf_fast<7, 0x89> (2);//alpgha
   gffast2 = gffast1;
   gffast3 = gffast1;

   for (int loop1 = -20; loop1 < 20; loop1++)
      {
      gf3 = gf3 * gf1;
      gffast3 = gffast3 * gffast1;
      gffast4 = gffast2 .power(loop1);
      assert(gf3 == gffast3);
      if (loop1 < 0)
         {
         power = (loop1 % non_zero);
         if (power < 0)
            power += non_zero;
         gf2 = gf1.inverse();
         gf4 = gf<7, 0x89> (1);
         for (int loop2 = loop1; loop2 < 0; loop2++)
            gf4 = gf4 * gf2;
         assert((gffast4 == gf4) && gffast4.log_gf() == power);
         }
      else if (loop1 == 0)
         {
         assert((1 == gffast4) && (gffast4.log_gf() == 0));
         }
      else
         {
         power = (loop1 % non_zero);
         gf4 = gf<7, 0x89> (1);
         gf2 = gf1;
         for (int loop2 = 0; loop2 < loop1; loop2++)
            gf4 = gf4 * gf2;
         assert((gffast4 == gf4) && gffast4.log_gf() == power);
         }
      }

   cout << "Checking addition, subtraction,multiplication and division" << std::endl;
   //check addition, subtraction, multiplication and division
   int tmp_val;
   for (int loop1 = 0; loop1 < num_of_elements; loop1++)
      {
      for (int loop2 = 0; loop2 < num_of_elements; loop2++)
         {
         gf1 = gf<7, 0x89> (loop1);
         gf2 = gf<7, 0x89> (loop2);

         gffast1 = gf_fast<7, 0x89> (loop1);
         gffast2 = gf_fast<7, 0x89> (loop2);

         //addition
         gf3 = gf1 + gf2;
         gffast3 = gffast1 + gffast2;
         assert(gf3 == gffast3);
         tmp_val = gf3;
         gffast4 = gf_fast<7, 0x89> (tmp_val);
         assert(gffast4.log_gf() == gffast3.log_gf());

         //subtraction
         gf3 = gf1 - gf2;
         gffast3 = gffast1 - gffast2;
         assert(gf3 == gffast3);
         tmp_val = gf3;
         gffast4 = gf_fast<7, 0x89> (tmp_val);
         assert(gffast4.log_gf() == gffast3.log_gf());

         //multiplication
         gf3 = gf1 * gf2;
         gffast3 = gffast1 * gffast2;
         assert(gf3 == gffast3);
         tmp_val = gf3;
         gffast4 = gf_fast<7, 0x89> (tmp_val);
         assert(gffast4.log_gf() == gffast3.log_gf());

         //division
         if (loop2 != 0) //don't divide by 0!
            {
            gf3 = gf1 / gf2;
            gffast3 = gffast1 / gffast2;
            assert(gf3 == gffast3);
            tmp_val = gf3;
            gffast4 = gf_fast<7, 0x89> (tmp_val);
            assert(gffast4.log_gf() == gffast3.log_gf());
            }
         }
      }

   cout << "Checking inverses" << std::endl;
   //check the inverses
   for (int loop1 = 1; loop1 < num_of_elements; loop1++)
      {
      gf1 = gf<7, 0x89> (loop1);
      gffast1 = gf_fast<7, 0x89> (loop1);
      gf2 = gf1.inverse();
      gffast2 = gffast1.inverse();
      tmp_val = gf2;
      gffast3 = gf_fast<7, 0x89> (tmp_val);
      assert((gf2 == gffast2) && (gffast2.log_gf() == gffast3.log_gf()));
      }

   cputimer t1;
   cout << "Proving that gf_fast is faster at multiplication:" << std::endl;
   cout
         << "multiplying all non-zero field elements together 1,000,000 times in gf<7, 0x89>" << std::endl;
   t1.start();
   gf1 = gf<7, 0x89> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gf1 = gf1 * gf<7, 0x89> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;

   cout
         << "multiplying all non-zero field elements together 1,000,000 times in gf_fast<7, 0x89>" << std::endl;
   t1.start();
   gffast1 = gf_fast<7, 0x89> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gffast1 = gffast1 * gf_fast<7, 0x89> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;

   cout << "Proving that gf_fast is as fast at additions as gf:" << std::endl;
   cout
         << "adding all non-zero field elements together 1,000,000 times in gf" << std::endl;
   t1.start();
   gf1 = gf<7, 0x89> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gf1 = gf1 + gf<7, 0x89> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;

   cout
         << "adding all non-zero field elements together 1,000,000 times in gf_fast" << std::endl;
   t1.start();
   gffast1 = gf_fast<7, 0x89> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gffast1 = gffast1 + gf_fast<7, 0x89> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;

   }

void TestFastGF256()
   {
   gf<8, 0x11D> gf1, gf2, gf3, gf4;
   gf_fast<8, 0x11D> gffast1, gffast2, gffast3, gffast4;
   int non_zero = 255;
   int num_of_elements = 256;
   int power = 0;

   cout << std::endl << "Checking gf_fast<8, 0x11D> against gf<8, 0x11D>" << std::endl;
   cout << "Checking the powers" << std::endl;
   //check the powers
   gf1 = gf<8, 0x11D> (2);//alpha
   gf3 = gf1;
   gffast1 = gf_fast<8, 0x11D> (2);//alpgha
   gffast2 = gffast1;
   gffast3 = gffast1;

   for (int loop1 = -20; loop1 < 20; loop1++)
      {
      gf3 = gf3 * gf1;
      gffast3 = gffast3 * gffast1;
      gffast4 = gffast2 .power(loop1);
      assert(gf3 == gffast3);
      if (loop1 < 0)
         {
         power = (loop1 % non_zero);
         if (power < 0)
            power += non_zero;
         gf2 = gf1.inverse();
         gf4 = gf<8, 0x11D> (1);
         for (int loop2 = loop1; loop2 < 0; loop2++)
            gf4 = gf4 * gf2;
         assert((gffast4 == gf4) && gffast4.log_gf() == power);
         }
      else if (loop1 == 0)
         {
         assert((1 == gffast4) && (gffast4.log_gf() == 0));
         }
      else
         {
         power = (loop1 % non_zero);
         gf4 = gf<8, 0x11D> (1);
         gf2 = gf1;
         for (int loop2 = 0; loop2 < loop1; loop2++)
            gf4 = gf4 * gf2;
         assert((gffast4 == gf4) && gffast4.log_gf() == power);
         }
      }

   cout << "Checking addition, subtraction,multiplication and division" << std::endl;
   //check addition, subtraction, multiplication and division
   int tmp_val;
   for (int loop1 = 0; loop1 < num_of_elements; loop1++)
      {
      for (int loop2 = 0; loop2 < num_of_elements; loop2++)
         {
         gf1 = gf<8, 0x11D> (loop1);
         gf2 = gf<8, 0x11D> (loop2);

         gffast1 = gf_fast<8, 0x11D> (loop1);
         gffast2 = gf_fast<8, 0x11D> (loop2);

         //addition
         gf3 = gf1 + gf2;
         gffast3 = gffast1 + gffast2;
         assert(gf3 == gffast3);
         tmp_val = gf3;
         gffast4 = gf_fast<8, 0x11D> (tmp_val);
         assert(gffast4.log_gf() == gffast3.log_gf());

         //subtraction
         gf3 = gf1 - gf2;
         gffast3 = gffast1 - gffast2;
         assert(gf3 == gffast3);
         tmp_val = gf3;
         gffast4 = gf_fast<8, 0x11D> (tmp_val);
         assert(gffast4.log_gf() == gffast3.log_gf());

         //multiplication
         gf3 = gf1 * gf2;
         gffast3 = gffast1 * gffast2;
         assert(gf3 == gffast3);
         tmp_val = gf3;
         gffast4 = gf_fast<8, 0x11D> (tmp_val);
         assert(gffast4.log_gf() == gffast3.log_gf());

         //division
         if (loop2 != 0) //don't divide by 0!
            {
            gf3 = gf1 / gf2;
            gffast3 = gffast1 / gffast2;
            assert(gf3 == gffast3);
            tmp_val = gf3;
            gffast4 = gf_fast<8, 0x11D> (tmp_val);
            assert(gffast4.log_gf() == gffast3.log_gf());
            }
         }
      }

   cout << "Checking inverses" << std::endl;
   //check the inverses
   for (int loop1 = 1; loop1 < num_of_elements; loop1++)
      {
      gf1 = gf<8, 0x11D> (loop1);
      gffast1 = gf_fast<8, 0x11D> (loop1);
      gf2 = gf1.inverse();
      gffast2 = gffast1.inverse();
      tmp_val = gf2;
      gffast3 = gf_fast<8, 0x11D> (tmp_val);
      assert((gf2 == gffast2) && (gffast2.log_gf() == gffast3.log_gf()));
      }

   cputimer t1;
   cout << "Proving that gf_fast is faster at multiplication:" << std::endl;
   cout
         << "multiplying all non-zero field elements together 1,000,000 times in gf<8, 0x11D>" << std::endl;
   t1.start();
   gf1 = gf<8, 0x11D> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gf1 = gf1 * gf<8, 0x11D> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;

   cout
         << "multiplying all non-zero field elements together 1,000,000 times in gf_fast<8, 0x11D>" << std::endl;
   t1.start();
   gffast1 = gf_fast<8, 0x11D> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gffast1 = gffast1 * gf_fast<8, 0x11D> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;

   cout << "Proving that gf_fast is as fast at additions as gf:" << std::endl;
   cout
         << "adding all non-zero field elements together 1,000,000 times in gf" << std::endl;
   t1.start();
   gf1 = gf<8, 0x11D> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gf1 = gf1 + gf<8, 0x11D> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;

   cout
         << "adding all non-zero field elements together 1,000,000 times in gf_fast" << std::endl;
   t1.start();
   gffast1 = gf_fast<8, 0x11D> (1);
   for (int loop1 = 0; loop1 < 1000000; loop1++)
      for (int loop2 = 1; loop2 < non_zero; loop2++)
         gffast1 = gffast1 + gf_fast<8, 0x11D> (loop2);
   t1.stop();
   cout << "This took " << t1 << std::endl;

   }

//helper function to generate the look up table required by gf_fast
void ProduceLookupTables()
   {
   /*
    template class gf<9, 0x211> ; // 1 { 0 0001 0001 }
    template class gf<10, 0x409> ; // 1 { 00 0000 1001 }
    */
   gf<8, 0x11D> alpha(2);
   gf<8, 0x11D> pow_of_alpha(1);
   const int num_of_elements = 256;
   int non_zero = num_of_elements - 1;

   int pow_lut[num_of_elements];
   int log_lut[num_of_elements];

   log_lut[0] = 0; //convention;
   pow_lut[non_zero] = 1;
   for (int loop = 0; loop < non_zero; loop++)
      {
      pow_lut[loop] = pow_of_alpha;
      log_lut[pow_of_alpha] = loop;
      pow_of_alpha *= alpha;
      }
   cout << std::endl << "template <> const int gf_fast<8, 0x11D>::log_lut[256] = {";
   for (int loop = 0; loop < non_zero; loop++)
      {
      cout << log_lut[loop] << ", ";
      }
   cout << log_lut[non_zero] << "};" << std::endl;

   cout << std::endl << "template <> const int gf_fast<8, 0x11D>::pow_lut[256] = {";
   for (int loop = 0; loop < non_zero; loop++)
      {
      cout << pow_lut[loop] << ", ";
      }
   cout << pow_lut[non_zero] << "};" << std::endl;

   }

/*!
 * \brief Test program for GF class
 * \author  Johann Briffa
 */

int main(int argc, char *argv[])
   {
   TestBinaryField();
   TestRijndaelField();
   ListField<2, 0x7> ();
   ListField<3, 0xB> ();
   ListField<4, 0x13> ();
   TestMulDiv<3, 0xB> ();
   TestGenPowerGF2();
   TestGenPowerGF8();
   // TODO: templatize tests for gf_fast
   TestFastGF2();
   TestFastGF4();
   TestFastGF8();
   TestFastGF16();
   TestFastGF32();
   TestFastGF64();
   TestFastGF128();
   TestFastGF256();
   //ProduceLookupTables();
   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return testgf::main(argc, argv);
   }
