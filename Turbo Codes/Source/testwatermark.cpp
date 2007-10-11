#include "watermarkcode.h"
#include "logrealfast.h"

#include <iostream>
using namespace std;

int main(int argc, char *argv[])
   {
   using libcomm::watermarkcode;
   using libbase::logrealfast;
   using std::cout;
   using std::cerr;

   int N=100, n=5, k=3;
   if(argc < 3)
      cerr << "Usage: " << argv[0] << " [<n> <k>]\n";
   else
      {
      n = atoi(argv[1]);
      k = atoi(argv[2]);
      }
   watermarkcode<logrealfast> c(N,n,k,0);
   cout << c.description() << "\n";
   }
