/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "lut_interleaver.h"
#include <stdlib.h>
#include <iostream>

namespace libcomm {

// static members

const int lut_interleaver::tail = -1;

// transform functions

void lut_interleaver::transform(const libbase::vector<int>& in, libbase::vector<int>& out) const
   {
   const int tau = lut.size();
   assertalways(in.size() == tau);
   out.init(in);
   for(int t=0; t<tau; t++)
      if(lut(t) == tail)
         out(t) = fsm::tail;
      else
         out(t) = in(lut(t));
   }

void lut_interleaver::transform(const libbase::matrix<double>& in, libbase::matrix<double>& out) const
   {
   const int tau = lut.size();
   const int K = in.ysize();
   assertalways(in.xsize() == tau);
   out.init(in);
   for(int t=0; t<tau; t++)
      if(lut(t) == tail)
         for(int i=0; i<K; i++)
            out(t, i) = 1.0/double(K);
      else
         for(int i=0; i<K; i++)
            out(t, i) = in(lut(t), i);
   }

void lut_interleaver::inverse(const libbase::matrix<double>& in, libbase::matrix<double>& out) const
   {
   const int tau = lut.size();
   const int K = in.ysize();
   assertalways(in.xsize() == tau);
   out.init(in);
   for(int t=0; t<tau; t++)
      if(lut(t) == tail)
         for(int i=0; i<K; i++)
            out(t, i) = 1.0/double(K);
      else
         for(int i=0; i<K; i++)
            out(lut(t), i) = in(t, i);
   }

// additional matrix types for transform/inverse

void lut_interleaver::transform(const libbase::matrix<libbase::logrealfast>& in, libbase::matrix<libbase::logrealfast>& out) const
   {
   const int tau = lut.size();
   const int K = in.ysize();
   assertalways(in.xsize() == tau);
   out.init(in);
   for(int t=0; t<tau; t++)
      if(lut(t) == tail)
         for(int i=0; i<K; i++)
            out(t, i) = 1.0/double(K);
      else
         for(int i=0; i<K; i++)
            out(t, i) = in(lut(t), i);
   }

void lut_interleaver::inverse(const libbase::matrix<libbase::logrealfast>& in, libbase::matrix<libbase::logrealfast>& out) const
   {
   const int tau = lut.size();
   const int K = in.ysize();
   assertalways(in.xsize() == tau);
   out.init(in);
   for(int t=0; t<tau; t++)
      if(lut(t) == tail)
         for(int i=0; i<K; i++)
            out(t, i) = 1.0/double(K);
      else
         for(int i=0; i<K; i++)
            out(lut(t), i) = in(t, i);
   }

}; // end namespace
