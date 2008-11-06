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

// Interface with mapper

void map_stipple::advance() const
   {
   assertalways(tau > 0);
   assertalways(sets > 0);
   // check if matrix is already set
   if(pattern.size() == tau*(sets+1))
      return;
   // initialise the pattern matrix
   pattern.init(tau*(sets+1));
   for(int i=0, t=0; t<tau; t++)
      for(int s=0; s<=sets; s++, i++)
         pattern(i) = (s==0 || (s-1)==t%sets);
   }

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

void map_stipple::doinverse(const libbase::vector< libbase::vector<double> >& pin, libbase::vector< libbase::vector<double> >& pout) const
   {
   assertalways(pin.size()==tau*2);
   assertalways(pin(0).size()==M);
   // final matrix size depends on the number of set positions
   libbase::vector< libbase::vector<double> > ptable;
   ptable.init(pattern.size());
   for(int i=0; i<pattern.size(); i++)
      ptable(i).init(M);
   // invert the puncturing
   for(int i=0, ii=0; i<pattern.size(); i++)
      if(pattern(i))
         {
         for(int j=0; j<M; j++)
            ptable(i)(j) = pin(ii)(j);
         ii++;
         }
      else
         {
         for(int j=0; j<M; j++)
            ptable(i)(j) = 1.0/M;
         }
   // do the base (straight) inverse mapping
   map_straight::doinverse(ptable, pout);
   }

// Description

std::string map_stipple::description() const
   {
   std::ostringstream sout;
   sout << "Stipple Mapper (" << sets << ")";
   return sout.str();
   }

// Serialization Support

std::ostream& map_stipple::serialize(std::ostream& sout) const
   {
   map_straight::serialize(sout);
   sout << sets << "\n";
   return sout;
   }

std::istream& map_stipple::serialize(std::istream& sin)
   {
   map_straight::serialize(sin);
   sin >> sets;
   return sin;
   }

}; // end namespace
