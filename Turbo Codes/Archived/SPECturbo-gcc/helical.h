#ifndef __helical_h
#define __helical_h

#include "config.h"
#include "vcs.h"

#include "lut_interleaver.h"

extern const vcs helical_version;

class helical : public virtual lut_interleaver {
   int rows, cols;
public:
   helical(const int tau, const int rows, const int cols);
   ~helical();

   void print(ostream& s) const;
};

// Template functions

inline helical::helical(const int tau, const int rows, const int cols)
   {
   helical::rows = rows;
   helical::cols = cols;

   int blklen = rows*cols;
   if(blklen > tau)
      {
      cerr << "FATAL ERROR (helical): Interleaver block size cannot be greater than BCJR block.\n";
      exit(1);
      }
   lut.init(tau);
   int row = rows-1, col = 0;
   for(int i=0; i<blklen; i++)
      {
      lut(i) = row*cols + col;
      row = (row-1+rows) % rows;
      col = (col+1) % cols;
      }
   for(int i=blklen; i<tau; i++)
      lut(i) = i;
   }

inline helical::~helical()
   {
   }
   
#endif

