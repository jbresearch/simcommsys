#include "logrealfast.h"
#include "watermarkcode.h"
#include "bsid.h"

#include <iostream>

int main(int argc, char *argv[])
   {
   using std::cout;
   using std::cerr;

   // common parameters
   const int I=10, xmax=50;
   const int n=5, k=3;

   // create a watermark codec
   using libbase::logrealfast;
   using libcomm::watermarkcode;
   watermarkcode<logrealfast> modem(n,k,0, I,xmax);
   cout << modem.description() << "\n";
   
   return 0;
   }
