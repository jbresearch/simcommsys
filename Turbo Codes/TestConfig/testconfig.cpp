/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "config.h"
#include "matrix.h"
#include "multi_array.h"

#include <boost/lambda/lambda.hpp>
#include <iterator>
#include <algorithm>

#include <sstream>
#include <iostream>

using libbase::matrix;
using std::cout;

void print_standard_sizes()
   {
   cout << '\n';
   cout << "Type:      \tSize (bits):\n";
   cout << "~~~~~      \t~~~~~~~~~~~~\n";

   cout << "char       \t" << sizeof(char)*8 << "\n";
   cout << "short      \t" << sizeof(short)*8 << "\n";
   cout << "int        \t" << sizeof(int)*8 << "\n";
   cout << "long       \t" << sizeof(long)*8 << "\n";
   cout << "long long  \t" << sizeof(long long)*8 << "\n";

   cout << "float      \t" << sizeof(float)*8 << "\n";
   cout << "double     \t" << sizeof(double)*8 << "\n";
   cout << "long double\t" << sizeof(long double)*8 << "\n";
   }

void print_new_sizes()
   {
   cout << '\n';
   cout << "Type:      \tSize (bits):\n";
   cout << "~~~~~      \t~~~~~~~~~~~~\n";

   cout << "int8u      \t" << sizeof(libbase::int8u)*8 << "\n";
   cout << "int16u     \t" << sizeof(libbase::int16u)*8 << "\n";
   cout << "int32u     \t" << sizeof(libbase::int32u)*8 << "\n";
   cout << "int64u     \t" << sizeof(libbase::int64u)*8 << "\n";

   cout << "int8s      \t" << sizeof(libbase::int8s)*8 << "\n";
   cout << "int16s     \t" << sizeof(libbase::int16s)*8 << "\n";
   cout << "int32s     \t" << sizeof(libbase::int32s)*8 << "\n";
   cout << "int64s     \t" << sizeof(libbase::int64s)*8 << "\n";
   }

void testmatrixmul()
   {
   cout << "\nMatrix Multiplication:\n\n";
   matrix<int> A(3,2);
   A(0,0) = 1;
   A(1,0) = 0;
   A(2,0) = 2;
   A(0,1) = -1;
   A(1,1) = 3;
   A(2,1) = 1;
   matrix<int> B(2,3);
   B(0,0) = 3;
   B(1,0) = 1;
   B(0,1) = 2;
   B(1,1) = 1;
   B(0,2) = 1;
   B(1,2) = 0;
   cout << "A = " << A;
   cout << "B = " << B;
   cout << "AB = " << A*B;
   }

void testmatrixinv()
   {
   cout << "\nMatrix Inversion:\n\n";
   matrix<int> A(3,3);
   A(0,0) = 1;
   A(1,0) = 0;
   A(2,0) = -2;
   A(0,1) = 4;
   A(1,1) = 1;
   A(2,1) = 0;
   A(0,2) = 1;
   A(1,2) = 1;
   A(2,2) = 7;
   cout << "A = " << A;
   cout << "inv(A) = " << A.inverse();
   cout << "inv(A).A = " << A.inverse()*A;
   }

void testboost_foreach(const std::string& s)
   {
   using namespace boost::lambda;
   typedef std::istream_iterator<int> in;
   std::istringstream sin(s);

   std::cout << "\nBoost ForEach Test:\n\n";
   std::for_each(
      in(sin), in(), std::cout << (_1 * 3) << " " );
   std::cout << "\n";
   }

void testboost_array()
   {
   // Constants
   const int xmax = 5;
   const int tau = 50;
   const int I = 2;
   // Set up forward matrix
   typedef boost::multi_array<double,2> array2d_t;
   typedef boost::multi_array_types::extent_range range;
   array2d_t F(boost::extents[tau+1][range(-xmax,xmax+1)]);

   std::cout << "\nBoost MultiArray Test:\n\n";
   // Initial conditions
   //F = 0;
   F[0][0] = 1;
   // compute remaining matrix values
   typedef array2d_t::index index;
   for(index j=1; j<=tau; ++j)
      {
      const index amin = std::max<index>(-xmax,1-j);
      const index amax = xmax;
      for(index a=amin; a<=amax; ++a)
         {
         const index ymin = std::max<index>(-xmax,a-1);
         const index ymax = std::min<index>(xmax,a+I);
         for(index y=ymin; y<=ymax; ++y)
            F[j][y] += F[j-1][a];
         }
      }
   // output results
   for(index x=-xmax; x<=xmax; ++x)
      std::cout << x << "\t" << F[tau][x] << "\n";
   std::cout << "\n";
   }

template <class T>
void display_array(boost::multi_array<T,2>& A)
   {
   typedef boost::multi_array<T,2> array2_t;
   typedef boost::multi_array<T,1> array1_t;
   for(typename array2_t::iterator i = A.begin(); i != A.end(); ++i, std::cout << "\n")
      for(typename array1_t::iterator j = i->begin(); j != i->end(); ++j, std::cout << "\t")
         std::cout << *j;
   std::cout << "\n";
   }

void testboost_iterators()
   {
   std::cout << "\nBoost Iterator Usage Test:\n\n";
   boost::assignable_multi_array<double,2> A(boost::extents[3][4]);
   display_array(A);
   A = 1;
   display_array(A);
   }

int main()
   {
   print_standard_sizes();
   print_new_sizes();
   testmatrixmul();
   testmatrixinv();
   testboost_foreach("1 2 3\n");
   testboost_array();
   testboost_iterators();
   }
