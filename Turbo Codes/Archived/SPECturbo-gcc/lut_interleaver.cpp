#include "lut_interleaver.h"
             
#include <iostream.h>
#include <stdlib.h>
             
const vcs lut_interleaver_version("Lookup Table Interleaver module (lut_interleaver)", 1.10);

const int lut_interleaver::tail = -1;

void lut_interleaver::print(ostream& s) const
   {
   s << "Unknown LUT Interleaver";
   }