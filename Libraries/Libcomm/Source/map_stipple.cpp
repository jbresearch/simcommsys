/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "map_stipple.h"
#include <stdlib.h>
#include <sstream>

namespace libcomm {

// Serialization Support

const libbase::serializer map_stipple::shelper("mapper", "map_stipple", map_stipple::create);

// Internal functions

void map_stipple::init(int tau, int sets)
   {
   assertalways(tau > 0);
   assertalways(sets > 0);
   map_stipple::tau = tau;
   map_stipple::sets = sets;
   // initialise the pattern matrix
   pattern.init(tau*(sets+1));
   for(int i=0, t=0; t<tau; t++)
      for(int s=0; s<=sets; s++, i++)
         pattern(i) = (s==0 || (s-1)==t%sets);
   }

// Interface with mapper

void map_stipple::dotransform(const libbase::vector<int>& in, libbase::vector<int>& out) const
   {
   // do the base (straight) mapping into a temporary space
   libbase::vector<int> s;
   map_straight::dotransform(in, s);
   // final vector size depends on the number of set positions
   assertalways(s.size()==pattern.size());
   out.init(2*tau);
   // puncture the results
   for(int i=0, ii=0; i<s.size(); i++)
      if(pattern(i))
         out(ii++) = s(i);
   }

void map_stipple::doinverse(const libbase::matrix<double>& pin, libbase::matrix<double>& pout) const
   {
   // do the base (straight) mapping into a temporary space
   libbase::matrix<double> ptable;
   map_straight::doinverse(pin, ptable);
   // final matrix size depends on the number of set positions
   assertalways(ptable.xsize()==2*tau);
   pout.init(pattern.size(),ptable.ysize());
   // invert the puncturing
   for(int i=0, ii=0; i<pout.xsize(); i++)
      if(pattern(i))
         {
         for(int j=0; j<pout.ysize(); j++)
            pout(i,j) = ptable(ii,j);
         ii++;
         }
      else
         {
         for(int j=0; j<pout.ysize(); j++)
            pout(i,j) = 1.0/pout.ysize();
         }

   }

// Description

std::string map_stipple::description() const
   {
   std::ostringstream sout;
   sout << "Stipple Mapper (" << tau << "x" << sets << ")";
   return sout.str();
   }

// Serialization Support

std::ostream& map_stipple::serialize(std::ostream& sout) const
   {
   map_straight::serialize(sout);
   sout << tau << "\n";
   sout << sets << "\n";
   return sout;
   }

std::istream& map_stipple::serialize(std::istream& sin)
   {
   map_straight::serialize(sin);
   sin >> tau;
   sin >> sets;
   init(tau, sets);
   return sin;
   }

}; // end namespace
