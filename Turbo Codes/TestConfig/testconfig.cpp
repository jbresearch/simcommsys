/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include <iostream>
using namespace std;

int main()
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
