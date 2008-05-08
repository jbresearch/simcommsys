/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "puncture.h"

namespace libcomm {

using std::cerr;

using libbase::vector;
using libbase::matrix;

// initialization function

void puncture::init(const matrix<bool>& pattern)
   {
   int t, s, k;
   // size info
   const int tau = pattern.xsize();
   const int sets = pattern.ysize();
   // work out the number of symbols that will be transmitted
   inputs = tau*sets;
   outputs = 0;
   for(t=0; t<tau; t++)
      for(s=0; s<sets; s++)
         if(pattern(t,s))
            outputs++;
   // now fill in the inverse-transform position vector
   pos.init(outputs);
   for(t=0, k=0; t<tau; t++)
      for(s=0; s<sets; s++)
         if(pattern(t,s))
            pos(k++) = t*sets+s;
   }

// puncturing / unpuncturing functions

void puncture::transform(const vector<sigspace>& in, vector<sigspace>& out) const
   {
   // validate input size
   if(in.size() != inputs)
      {
      cerr << "FATAL ERROR (puncture): size mismatch, transform input vector size " << in.size() << " should be " << inputs<< ".\n";
      exit(1);
      }
   // initialize results vector
   out.init(outputs);
   // do the transform
   for(int i=0; i<outputs; i++)
      out(i) = in(pos(i));
   }

void puncture::inverse(const vector<sigspace>& in, vector<sigspace>& out) const
   {
   // validate input size
   if(in.size() != outputs)
      {
      cerr << "FATAL ERROR (puncture): size mismatch, inverse input vector size " << in.size() << " should be " << outputs << ".\n";
      exit(1);
      }
   // initialize results vector
   out.init(inputs);
   out = sigspace(0,0);
   // do the inverse transform
   for(int i=0; i<outputs; i++)
      out(pos(i)) = in(i);
   }

void puncture::inverse(const matrix<double>& in, matrix<double>& out) const
   {
   // validate input size
   if(in.xsize() != outputs)
      {
      cerr << "FATAL ERROR (puncture): size mismatch, inverse input matrix size " << in.xsize() << " should be " << outputs << ".\n";
      exit(1);
      }
   // initialize results matrix
   const int M = in.ysize();
   out.init(inputs, M);
   out = 1/double(M);
   // do the inverse transform
   for(int i=0; i<outputs; i++)
      for(int j=0; j<M; j++)
         out(pos(i),j) = in(i,j);
   }

}; // end namespace
