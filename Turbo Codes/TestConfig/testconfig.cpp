/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include <iostream>
#include <matrix.h>

using libbase::matrix;
using std::cout;

void printsizes()
   {
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

void testmatrixmul()
   {
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
   }

int main()
   {
   printsizes();
   testmatrixmul();
   testmatrixinv();
   }
