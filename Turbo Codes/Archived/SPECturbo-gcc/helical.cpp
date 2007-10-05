#include "helical.h"
             
#include <iostream.h>
#include <stdlib.h>
             
const vcs helical_version("Helical Interleaver module (helical)", 1.00);

void helical::print(ostream& s) const
   {
   s << "Helical " << rows << "x" << cols << " Interleaver";
   }
