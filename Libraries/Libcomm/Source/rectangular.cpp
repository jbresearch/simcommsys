/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "rectangular.h"
#include <stdlib.h>
#include <sstream>

namespace libcomm {

const libbase::vcs rectangular::version("Rectangular Interleaver module (rectangular)", 1.41);

const libbase::serializer rectangular::shelper("interleaver", "rectangular", rectangular::create);


// initialisation functions

void rectangular::init(const int tau, const int rows, const int cols)
   {
   rectangular::rows = rows;
   rectangular::cols = cols;

   int blklen = rows*cols;
   if(blklen > tau)
      {
      std::cerr << "FATAL ERROR (rectangular): Interleaver block size cannot be greater than BCJR block.\n";
      exit(1);
      }
   lut.init(tau);
   int row = 0, col = 0;
   int i;
   for(i=0; i<blklen; i++)
      {
      row = i % rows;
      col = i / rows;
      lut(i) = row*cols + col;
      }
   for(i=blklen; i<tau; i++)
      lut(i) = i;
   }
   
// description output

std::string rectangular::description() const
   {
   std::ostringstream sout;
   sout << "Rectangular " << rows << "x" << cols << " Interleaver";
   return sout.str();
   }

// object serialization - saving

std::ostream& rectangular::serialize(std::ostream& sout) const
   {
   sout << lut.size() << "\n";
   sout << rows << "\n";
   sout << cols << "\n";
   return sout;
   }

// object serialization - loading

std::istream& rectangular::serialize(std::istream& sin)
   {
   int tau;
   sin >> tau;
   sin >> rows;
   sin >> cols;
   init(tau, rows, cols);
   return sin;
   }

}; // end namespace
