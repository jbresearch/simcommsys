/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "helical.h"
#include <stdlib.h>
#include <sstream>

namespace libcomm {

const libbase::serializer helical::shelper("interleaver", "helical", helical::create);

// initialisation functions

void helical::init(const int tau, const int rows, const int cols)
   {
   helical::rows = rows;
   helical::cols = cols;

   int blklen = rows*cols;
   if(blklen > tau)
      {
      std::cerr << "FATAL ERROR (helical): Interleaver block size cannot be greater than BCJR block.\n";
      exit(1);
      }
   lut.init(tau);
   int row = rows-1, col = 0;
   int i;
   for(i=0; i<blklen; i++)
      {
      lut(i) = row*cols + col;
      row = (row-1+rows) % rows;
      col = (col+1) % cols;
      }
   for(i=blklen; i<tau; i++)
      lut(i) = i;
   }

// description output

std::string helical::description() const
   {
   std::ostringstream sout;
   sout << "Helical " << rows << "x" << cols << " Interleaver";
   return sout.str();
   }

// object serialization - saving

std::ostream& helical::serialize(std::ostream& sout) const
   {
   sout << lut.size() << "\n";
   sout << rows << "\n";
   sout << cols << "\n";
   return sout;
   }

// object serialization - loading

std::istream& helical::serialize(std::istream& sin)
   {
   int tau;
   sin >> tau;
   sin >> rows;
   sin >> cols;
   init(tau, rows, cols);
   return sin;
   }

}; // end namespace
