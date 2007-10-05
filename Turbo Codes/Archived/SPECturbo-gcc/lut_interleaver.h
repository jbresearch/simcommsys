#ifndef __lut_interleaver_h
#define __lut_interleaver_h

#include "config.h"
#include "vcs.h"

#include "interleaver.h"
#include "fsm.h"

extern const vcs lut_interleaver_version;

/*
  Version 1.10 (29 Aug 1999)
  introduced concept of forced tail interleavers (as in divs95)
*/
class lut_interleaver : public virtual interleaver {
protected:
   static const int tail; // a special LUT entry to signify a forced tail
   vector<int> lut;
public:
   void transform(vector<int>& in, vector<int>& out) const;
   void transform(matrix<double>& in, matrix<double>& out) const;
   void inverse(matrix<double>& in, matrix<double>& out) const;

   virtual void print(ostream& s) const;
};

// Template functions

inline void lut_interleaver::transform(vector<int>& in, vector<int>& out) const
   {
   const int tau = lut.size();
   if(in.size() != tau || out.size() != tau)
      {
      cerr << "FATAL ERROR (lut_interleaver): vectors must have same size as LUT (in=" << in.size() << ", out=" << out.size() << ", lut=" << tau << ").\n";
      exit(1);
      }
   for(int t=0; t<tau; t++)
      if(lut(t) == tail)
         out(t) = fsm::tail;
      else
         out(t) = in(lut(t));
   }   

inline void lut_interleaver::transform(matrix<double>& in, matrix<double>& out) const
   {
   const int tau = lut.size();
   if(in.x_size() != tau || out.x_size() != tau)
      {
      cerr << "FATAL ERROR (lut_interleaver): matrices must have same x-size as LUT (in=" << in.x_size() << ", out=" << out.x_size() << ", lut=" << tau << ").\n";
      exit(1);
      }
   const int K = in.y_size();
   if(in.y_size() != out.y_size())
      {
      cerr << "FATAL ERROR (lut_interleaver): matrices must have same y-size (in=" << in.y_size() << ", out=" << out.y_size() << ").\n";
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

inline void lut_interleaver::inverse(matrix<double>& in, matrix<double>& out) const
   {
   const int tau = lut.size();
   if(in.x_size() != tau || out.x_size() != tau)
      {
      cerr << "FATAL ERROR (lut_interleaver): matrices must have same x-size as LUT (in=" << in.x_size() << ", out=" << out.x_size() << ", lut=" << tau << ").\n";
      exit(1);
      }
   const int K = in.y_size();
   if(in.y_size() != out.y_size())
      {
      cerr << "FATAL ERROR (lut_interleaver): matrices must have same y-size (in=" << in.y_size() << ", out=" << out.y_size() << ").\n";
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
   
#endif

