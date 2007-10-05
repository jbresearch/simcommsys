#include "diffturbo.h"
#include <sstream>
#include <stdio.h>

namespace libcomm {

using libbase::vector;
using libbase::matrix;

// initialization

template <class real> void diffturbo<real>::init()
   {
   using std::cerr;
   if(turbo<real>::tail_length() == 0)  // not sure if this is the same as endatzero
      {
      cerr << "FATAL ERROR (diffturbo): Can only handle unterminated systems.\n";
      exit(1);
      }
   if(turbo<real>::block_size() % 4 != 0)
      {
      cerr << "FATAL ERROR (diffturbo): Can only handle block sizes that are a multiple of 4.\n";
      exit(1);
      }
   if(turbo<real>::num_inputs() > 2)
      {
      cerr << "FATAL ERROR (diffturbo): Only binary input codes are handled by this implementation.\n";
      exit(1);
      }
   }

// constructor / destructor

template <class real> diffturbo<real>::diffturbo(const char *filename, fsm& encoder, const int tau, vector<interleaver *>& inter, const int iter, const bool simile, const bool endatzero, const bool parallel) : \
   turbo<real>(encoder, tau, inter, iter, simile, endatzero, parallel)
   {
   init();
   load_lut(filename, tau);
   }

// internal helper functions

template <class real> void diffturbo<real>::load_lut(const char *filename, const int tau)
   {
   diffturbo<real>::filename = filename;
   lut.init(tau);

   char buf[256];
   FILE *file = fopen(filename, "rb");
   if(file == NULL)
      {
      std::cerr << "FATAL ERROR (diffturbo): Cannot open LUT file (" << filename << ").\n";
      exit(1);
      }
   for(int i=0; i<tau; i++)
      {
      do {
         fscanf(file, "%[^\n]\n", buf);
         } while(buf[0] == '#');
      int x;
      sscanf(buf, "%d", &x);
      lut(i) = x;
      }
   fclose(file);
   }

template <class real> void diffturbo<real>::add(matrix<double>& z, matrix<double>& x, matrix<double>& y, int zp, int xp, int yp)
   {
   z(zp,0) = x(xp,0) * y(yp,0) + x(xp,1) * y(yp,1);
   z(zp,1) = x(xp,1) * y(yp,0) + x(xp,0) * y(yp,1);
   }

// codec functions

template <class real> void diffturbo<real>::encode(vector<int>& source, vector<int>& encoded)
   {
   int i;
   // Allocate memory for sources
   const int tau = turbo<real>::block_size();
   const int K = turbo<real>::num_inputs();
   source2.init(tau);
   source3.init(tau);
   encoded.init(tau);
   // First we sort the input vector in ascending order of confidence (get from LUT)
   for(i=0; i<tau; i++)
      source2(i) = source(lut(i));
   // Next we operate on the input vector an algebraic transformation (4-set division)
   const int x = tau/4;
   for(i=0; i<x; i++)
      {
      source3(0*x+i) = (source2(0*x+i) + source2(1*x+i) + source2(2*x+i)) % K;
      source3(1*x+i) = (source2(1*x+i) + source2(2*x+i) + source2(3*x+i)) % K;
      source3(2*x+i) = (source2(1*x+i) + source2(3*x+i)) % K;
      source3(3*x+i) = (source2(0*x+i) + source2(2*x+i)) % K;
      }
   // Then we pass the modified vector through the turbo encoder
   turbo<real>::encode(source3, encoded);
   }

template <class real> void diffturbo<real>::decode(vector<int>& decoded)
   {
   int i;
   // Allocate memory for results
   const int tau = turbo<real>::block_size();
   const int K = turbo<real>::num_inputs();
   decoded2.init(tau,K);
   decoded3.init(tau,K);
   decoded.init(tau);
   // First we decode using the standard turbo decoder
   turbo<real>::decode(decoded);
   // Then we compute the normalised statistics
   for(i=0; i<tau; i++)
      {
      double sum = turbo<real>::aposteriori(i,0) + turbo<real>::aposteriori(i,1);
      decoded2(i,0) = turbo<real>::aposteriori(i,0)/sum;
      decoded2(i,1) = turbo<real>::aposteriori(i,1)/sum;
      }
   // Next we operate on the probabilites the inverse algebraic transformation
   const int x = tau/4;
   for(i=0; i<x; i++)
      {
      // first we work out the 2-input additions:
      add(decoded3,decoded2,decoded2, 1*x+i, 0*x+i, 3*x+i);
      add(decoded3,decoded2,decoded2, 2*x+i, 1*x+i, 2*x+i);
      // then we work out the 3-input cases by using the 2-input results
      add(decoded3,decoded3,decoded2, 0*x+i, 2*x+i, 3*x+i);
      add(decoded3,decoded3,decoded2, 3*x+i, 1*x+i, 2*x+i);
      }
   // Finally we make a hard decision on the result (after inverting the sort)
   for(i=0; i<tau; i++)
      decoded(lut(i)) = (decoded3(i,1) > decoded3(i,0)) ? 1 : 0;
   }

// description output

template <class real> std::string diffturbo<real>::description() const
   {
   std::ostringstream sout;
   sout << "Diffused-Input ";
   sout << turbo<real>::description();
   return sout.str();
   }

// object serialization - saving

template <class real> std::ostream& diffturbo<real>::serialize(std::ostream& sout) const
   {
   sout << filename << "\n";
   sout << lut;
   turbo<real>::serialize(sout);
   return sout;
   }

// object serialization - loading

template <class real> std::istream& diffturbo<real>::serialize(std::istream& sin)
   {
   sin >> filename;
   sin >> lut;
   turbo<real>::serialize(sin);
   init();
   return sin;
   }

}; // end namespace

// Explicit Realizations

#include "mpreal.h"
#include "mpgnu.h"
#include "logreal.h"
#include "logrealfast.h"

namespace libcomm {

using libbase::mpreal;
using libbase::mpgnu;
using libbase::logreal;
using libbase::logrealfast;

using libbase::serializer;
using libbase::vcs;

#define VERSION 2.50

template class diffturbo<mpreal>;
template <> const serializer diffturbo<mpreal>::shelper = serializer("codec", "diffturbo<mpreal>", diffturbo<mpreal>::create);
template <> const vcs diffturbo<mpreal>::version = vcs("Diffused-Input Turbo Decoder module (diffturbo<mpreal>)", VERSION);

template class diffturbo<mpgnu>;
template <> const serializer diffturbo<mpgnu>::shelper = serializer("codec", "diffturbo<mpgnu>", diffturbo<mpgnu>::create);
template <> const vcs diffturbo<mpgnu>::version = vcs("Diffused-Input Turbo Decoder module (diffturbo<mpgnu>)", VERSION);

template class diffturbo<logreal>;
template <> const serializer diffturbo<logreal>::shelper = serializer("codec", "diffturbo<logreal>", diffturbo<logreal>::create);
template <> const vcs diffturbo<logreal>::version = vcs("Diffused-Input Turbo Decoder module (diffturbo<logreal>)", VERSION);

template class diffturbo<logrealfast>;
template <> const serializer diffturbo<logrealfast>::shelper = serializer("codec", "diffturbo<logrealfast>", diffturbo<logrealfast>::create);
template <> const vcs diffturbo<logrealfast>::version = vcs("Diffused-Input Turbo Decoder module (diffturbo<logrealfast>)", VERSION);

}; // end namespace
