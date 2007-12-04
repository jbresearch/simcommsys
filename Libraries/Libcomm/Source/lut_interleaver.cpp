#include "lut_interleaver.h"
#include <stdlib.h>
#include <iostream>

namespace libcomm {

const libbase::vcs lut_interleaver::version("Lookup Table Interleaver module (lut_interleaver)", 1.60);

// static members

const int lut_interleaver::tail = -1;

// transform functions

void lut_interleaver::transform(const libbase::vector<int>& in, libbase::vector<int>& out) const
   {
   const int tau = lut.size();
   if(in.size() != tau || out.size() != tau)
      {
      std::cerr << "FATAL ERROR (lut_interleaver): vectors must have same size as LUT (in=" << in.size() << ", out=" << out.size() << ", lut=" << tau << ").\n";
      exit(1);
      }
   for(int t=0; t<tau; t++)
      if(lut(t) == tail)
         out(t) = fsm::tail;
      else
         out(t) = in(lut(t));
   }   

void lut_interleaver::transform(const libbase::matrix<double>& in, libbase::matrix<double>& out) const
   {
   const int tau = lut.size();
   if(in.xsize() != tau || out.xsize() != tau)
      {
      std::cerr << "FATAL ERROR (lut_interleaver): matrices must have same x-size as LUT (in=" << in.xsize() << ", out=" << out.xsize() << ", lut=" << tau << ").\n";
      exit(1);
      }
   const int K = in.ysize();
   if(in.ysize() != out.ysize())
      {
      std::cerr << "FATAL ERROR (lut_interleaver): matrices must have same y-size (in=" << in.ysize() << ", out=" << out.ysize() << ").\n";
      exit(1);
      }
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
   if(in.xsize() != tau || out.xsize() != tau)
      {
      std::cerr << "FATAL ERROR (lut_interleaver): matrices must have same x-size as LUT (in=" << in.xsize() << ", out=" << out.xsize() << ", lut=" << tau << ").\n";
      exit(1);
      }
   const int K = in.ysize();
   if(in.ysize() != out.ysize())
      {
      std::cerr << "FATAL ERROR (lut_interleaver): matrices must have same y-size (in=" << in.ysize() << ", out=" << out.ysize() << ").\n";
      exit(1);
      }
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
   if(in.xsize() != tau || out.xsize() != tau)
      {
      std::cerr << "FATAL ERROR (lut_interleaver): matrices must have same x-size as LUT (in=" << in.xsize() << ", out=" << out.xsize() << ", lut=" << tau << ").\n";
      exit(1);
      }
   const int K = in.ysize();
   if(in.ysize() != out.ysize())
      {
      std::cerr << "FATAL ERROR (lut_interleaver): matrices must have same y-size (in=" << in.ysize() << ", out=" << out.ysize() << ").\n";
      exit(1);
      }
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
   if(in.xsize() != tau || out.xsize() != tau)
      {
      std::cerr << "FATAL ERROR (lut_interleaver): matrices must have same x-size as LUT (in=" << in.xsize() << ", out=" << out.xsize() << ", lut=" << tau << ").\n";
      exit(1);
      }
   const int K = in.ysize();
   if(in.ysize() != out.ysize())
      {
      std::cerr << "FATAL ERROR (lut_interleaver): matrices must have same y-size (in=" << in.ysize() << ", out=" << out.ysize() << ").\n";
      exit(1);
      }
   for(int t=0; t<tau; t++)
      if(lut(t) == tail)
         for(int i=0; i<K; i++)
            out(t, i) = 1.0/double(K);
      else
         for(int i=0; i<K; i++)
            out(lut(t), i) = in(t, i);
   }

}; // end namespace
