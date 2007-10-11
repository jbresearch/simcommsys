#include "watermarkcode.h"
#include "logrealfast.h"

#include <iostream>
using namespace std;

int main()
   {
   using libcomm::watermarkcode;
   using libbase::logrealfast;
   watermarkcode<logrealfast> c(100,5,5,0);
   }
