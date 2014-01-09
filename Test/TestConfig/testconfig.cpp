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

#include "config.h"
#include "matrix.h"
#include "multi_array.h"

#include <boost/lambda/lambda.hpp>
#include <iterator>
#include <algorithm>

#include <sstream>
#include <iostream>
#include <iomanip>

namespace testconfig {

using libbase::vector;
using libbase::matrix;
using std::cout;

void print_whitespace_test()
   {
   cout << std::endl;
   cout << "Char:       \tWhitespace?:" << std::endl;
   cout << "~~~~~       \t~~~~~~~~~~~~" << std::endl;

   cout << "\\r         \t" << (isspace('\r') ? "yes" : "no") << std::endl;
   cout << "\\n         \t" << (isspace('\n') ? "yes" : "no") << std::endl;
   cout << "\\t         \t" << (isspace('\t') ? "yes" : "no") << std::endl;
   }

void print_standard_limits()
   {
   cout << std::endl;
   cout << "Type:      \tValue:" << std::endl;
   cout << "~~~~~      \t~~~~" << std::endl;

   cout << "epsilon (f)\t" << std::numeric_limits<float>::epsilon()
         << std::endl;
   cout << "epsilon (d)\t" << std::numeric_limits<double>::epsilon()
         << std::endl;
   cout << "epsilon (ld)\t" << std::numeric_limits<long double>::epsilon()
         << std::endl;
   }

void print_standard_sizes()
   {
   cout << std::endl;
   cout << "Type:      \tSize (bits):" << std::endl;
   cout << "~~~~~      \t~~~~~~~~~~~~" << std::endl;

   cout << "bool       \t" << sizeof(bool) * 8 << std::endl;
   cout << "char       \t" << sizeof(char) * 8 << std::endl;
   cout << "short      \t" << sizeof(short) * 8 << std::endl;
   cout << "int        \t" << sizeof(int) * 8 << std::endl;
   cout << "long       \t" << sizeof(long) * 8 << std::endl;
   cout << "long long  \t" << sizeof(long long) * 8 << std::endl;

   cout << "float      \t" << sizeof(float) * 8 << std::endl;
   cout << "double     \t" << sizeof(double) * 8 << std::endl;
   cout << "long double\t" << sizeof(long double) * 8 << std::endl;

   cout << "void*      \t" << sizeof(void*) * 8 << std::endl;
   cout << "size_t     \t" << sizeof(size_t) * 8 << std::endl;
   }

void print_new_sizes()
   {
   cout << std::endl;
   cout << "Type:      \tSize (bits):" << std::endl;
   cout << "~~~~~      \t~~~~~~~~~~~~" << std::endl;

   cout << "int8u      \t" << sizeof(libbase::int8u) * 8 << std::endl;
   cout << "int16u     \t" << sizeof(libbase::int16u) * 8 << std::endl;
   cout << "int32u     \t" << sizeof(libbase::int32u) * 8 << std::endl;
   cout << "int64u     \t" << sizeof(libbase::int64u) * 8 << std::endl;
#if defined(USE_128BIT_INT)
   cout << "int128u    \t" << sizeof(libbase::int128u) * 8 << std::endl;
#endif

   cout << "int8s      \t" << sizeof(libbase::int8s) * 8 << std::endl;
   cout << "int16s     \t" << sizeof(libbase::int16s) * 8 << std::endl;
   cout << "int32s     \t" << sizeof(libbase::int32s) * 8 << std::endl;
   cout << "int64s     \t" << sizeof(libbase::int64s) * 8 << std::endl;
#if defined(USE_128BIT_INT)
   cout << "int128s    \t" << sizeof(libbase::int128s) * 8 << std::endl;
#endif
   }

void print_struct_sizes()
   {
   typedef struct {
      bool a :1;
      bool b :1;
      bool c :1;
      bool d :1;
   } struct_field_t;

   typedef struct {
      bool a;
      bool b;
      bool c;
      bool d;
   } struct_bool_t;

   cout << std::endl;
   cout << "Type:         \tSize (bits):" << std::endl;
   cout << "~~~~~         \t~~~~~~~~~~~~" << std::endl;

   cout << "struct_bool_t \t" << sizeof(struct_bool_t) * 8 << std::endl;
   cout << "struct_field_t\t" << sizeof(struct_field_t) * 8 << std::endl;
   }

void print_vector_sizes()
   {
   // create a vector with two elements
   vector<int> x;
   x.init(2);

   cout << std::endl;
   cout << "Type:      \tSize (bits):" << std::endl;
   cout << "~~~~~      \t~~~~~~~~~~~~" << std::endl;

   cout << "vector size type    \t" << sizeof(x.size()) * 8 << std::endl;
   cout << "vector container    \t" << sizeof(x) * 8 << std::endl;
   cout << "int elements        \t" << ((char*) &x(1) - (char*) &x(0)) * 8
         << std::endl;
   }

vector<int> makerangevector()
   {
   cout << std::endl << "Make vector range:" << std::endl << std::endl;
   // create space for result
   vector<int> x(10);
   cout << "Local address: " << &x << std::endl;
   cout << "Size: " << x.size() << std::endl;
   // init values
   for (int i = 0; i < x.size(); i++)
      x(i) = i;
   cout << "Contents: {";
   x.serialize(cout, ',');
   cout << "}" << std::endl;
   // return result
   return x;
   }

void accessvectorbyvalue(vector<int> x)
   {
   cout << std::endl << "Access vector by value:" << std::endl << std::endl;
   cout << "Address: " << &x << std::endl;
   cout << "Size: " << x.size() << std::endl;
   cout << "Contents: {";
   x.serialize(cout, ',');
   cout << "}" << std::endl;
   }

void accessvectorbyreference(vector<int>& x)
   {
   cout << std::endl << "Access vector by reference:" << std::endl << std::endl;
   cout << "Address: " << &x << std::endl;
   cout << "Size: " << x.size() << std::endl;
   cout << "Contents: {";
   x.serialize(cout, ',');
   cout << "}" << std::endl;
   }

void testvector()
   {
   // test default construction
   vector<int> x;
   assert(x.size() == 0);
   // test integer construction (two forms)
   vector<int> y1(10);
   assert(y1.size() == 10);
   vector<int> y2 = vector<int> (10);
   assert(y2.size() == 10);
   // test copy construction (two forms)
   vector<int> z1(y1);
   assert(z1.size() == 10);
   vector<int> z2 = y1;
   assert(z2.size() == 10);
   // test copy assignment
   x = y1;
   assert(x.size() == 10);
   // test vector return by value
   vector<int> r = makerangevector();
   cout << std::endl << "Test vector return by value:" << std::endl
         << std::endl;
   cout << "Address: " << &r << std::endl;
   cout << "Size: " << r.size() << std::endl;
   cout << "Contents: {";
   r.serialize(cout, ',');
   cout << "}" << std::endl;
   // test vector passing
   cout << std::endl << "Test vector access on:" << std::endl << std::endl;
   cout << "Address: " << &r << std::endl;
   cout << "Size: " << r.size() << std::endl;
   cout << "Contents: {";
   r.serialize(cout, ',');
   cout << "}" << std::endl;
   accessvectorbyreference(r);
   accessvectorbyvalue(r);
   }

void testmatrixmul()
   {
   cout << std::endl << "Matrix Multiplication:" << std::endl << std::endl;
   matrix<int> A(3, 2);
   A(0, 0) = 1;
   A(1, 0) = 0;
   A(2, 0) = 2;
   A(0, 1) = -1;
   A(1, 1) = 3;
   A(2, 1) = 1;
   matrix<int> B(2, 3);
   B(0, 0) = 3;
   B(1, 0) = 1;
   B(0, 1) = 2;
   B(1, 1) = 1;
   B(0, 2) = 1;
   B(1, 2) = 0;
   cout << "A = " << A;
   cout << "B = " << B;
   matrix<int> AB = A * B;
   cout << "AB = " << AB;
   // test for result values
   matrix<int> R(3, 3);
   R(0, 0) = 2;
   R(1, 0) = 3;
   R(2, 0) = 7;
   R(0, 1) = 1;
   R(1, 1) = 3;
   R(2, 1) = 5;
   R(0, 2) = 1;
   R(1, 2) = 0;
   R(2, 2) = 2;
   assert(AB.isequalto(R));
   }

void testmatrixinv()
   {
   cout << std::endl << "Matrix Inversion:" << std::endl << std::endl;
   matrix<int> A(3, 3);
   A(0, 0) = 1;
   A(1, 0) = 0;
   A(2, 0) = -2;
   A(0, 1) = 4;
   A(1, 1) = 1;
   A(2, 1) = 0;
   A(0, 2) = 1;
   A(1, 2) = 1;
   A(2, 2) = 7;
   cout << "A = " << A;
   matrix<int> Ainv = A.inverse();
   cout << "inv(A) = " << Ainv;
   matrix<int> R = Ainv * A;
   cout << "inv(A).A = " << R;
   assert(R.isequalto(matrix<int>::eye(3)));
   }

void testmatrixops()
   {
   cout << std::endl << "Matrix Operations:" << std::endl << std::endl;
   matrix<int> A;
   std::istringstream s("4 4\n2 4 1 3\n-1 -2 1 0\n0 0 2 2\n3 6 2 5");
   s >> A;
   cout << "A = " << A;
   matrix<int> Aref = A.reduce_to_ref();
   cout << "ref(A) = " << Aref;
   int r = A.rank();
   cout << "rank(A) = " << r << std::endl;
   assert(r == 2);
   }

void testboost_foreach(const std::string& s)
   {
   using namespace boost::lambda;
   typedef std::istream_iterator<int> in;
   std::istringstream sin(s);

   std::cout << std::endl << "Boost ForEach Test:" << std::endl << std::endl;
   std::for_each(in(sin), in(), std::cout << (_1 * 3) << " ");
   std::cout << std::endl;
   }

void testboost_array()
   {
   // Constants
   const int xmax = 5;
   const int tau = 50;
   const int I = 2;
   // Set up forward matrix
   typedef boost::multi_array<double, 2> array2d_t;
   typedef boost::multi_array_types::extent_range range;
   array2d_t F(boost::extents[tau + 1][range(-xmax, xmax + 1)]);

   std::cout << std::endl << "Boost MultiArray Test:" << std::endl << std::endl;
   // Initial conditions
   //F = 0;
   F[0][0] = 1;
   // compute remaining matrix values
   typedef array2d_t::index index;
   for (index j = 1; j <= tau; ++j)
      {
      const index amin = std::max<index>(-xmax, 1 - j);
      const index amax = xmax;
      for (index a = amin; a <= amax; ++a)
         {
         const index ymin = std::max<index>(-xmax, a - 1);
         const index ymax = std::min<index>(xmax, a + I);
         for (index y = ymin; y <= ymax; ++y)
            F[j][y] += F[j - 1][a];
         }
      }
   // output results
   for (index x = -xmax; x <= xmax; ++x)
      std::cout << x << "\t" << F[tau][x] << std::endl;
   std::cout << std::endl;
   }

template <class T>
void display_array(boost::multi_array<T, 2>& A)
   {
   typedef boost::multi_array<T, 2> array2_t;
   typedef boost::multi_array<T, 1> array1_t;
   for (typename array2_t::iterator i = A.begin(); i != A.end(); ++i, std::cout
         << std::endl)
      for (typename array1_t::iterator j = i->begin(); j != i->end(); ++j, std::cout
            << "\t")
         std::cout << *j;
   std::cout << std::endl;
   }

void testboost_iterators()
   {
   std::cout << std::endl << "Boost Iterator Usage Test:" << std::endl
         << std::endl;
   boost::assignable_multi_array<double, 2> A(boost::extents[3][4]);
   display_array(A);
   A = 1;
   display_array(A);
   }

void testbool_ops()
   {
   std::cout << std::endl << "Bool Operator Test:" << std::endl << std::endl;
   for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
         {
         const bool a = bool(i);
         const bool b = bool(j);
         const bool c = a + b;
         std::cout << a << " + " << b << " = " << c << std::endl;
         }
   std::cout << std::endl;
   for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
         {
         const bool a = bool(i);
         const bool b = bool(j);
         const bool c = a * b;
         std::cout << a << " * " << b << " = " << c << std::endl;
         }
   }

void test128bit()
   {
#if defined(USE_128BIT_INT)
   std::cout << std::endl << "128-bit Integer Test:" << std::endl << std::endl;
   libbase::int128u x = 0;
   for (int i = 0; i < 128/4; i++)
      {
      x <<= 4;
      x |= (i % 16);
      // show integer value in hex
      const std::ios::fmtflags flags = std::cout.flags();
      std::cout << "0x" << std::hex << std::setfill('0');
      std::cout << std::setw(16) << libbase::int64u((x>>64) & 0xffffffffffffffffL);
      std::cout << std::setw(16) << libbase::int64u(x & 0xffffffffffffffffL);
      std::cout << std::dec << std::endl;
      std::cout.flags(flags);
      }
#endif
   }

/*!
 * \brief   Test program for various base functions and facilities
 * \author  Johann Briffa
 */

int main(int argc, char *argv[])
   {
   print_whitespace_test();
   print_standard_limits();
   print_standard_sizes();
   print_new_sizes();
   print_struct_sizes();
   print_vector_sizes();
   testvector();
   testmatrixmul();
   testmatrixinv();
   testmatrixops();
   testboost_foreach("1 2 3\n");
   testboost_array();
   testboost_iterators();
   testbool_ops();
   test128bit();
   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return testconfig::main(argc, argv);
   }
