/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "map_permuted.h"
#include <stdlib.h>
#include <sstream>

namespace libcomm {

// Serialization Support

const libbase::serializer map_permuted::shelper("mapper", "map_permuted", map_permuted::create);

// Interface with mapper

void map_permuted::dotransform(const libbase::vector<int>& in, libbase::vector<int>& out) const
   {
   // do the base (straight) mapping into a temporary space
   libbase::vector<int> s;
   map_straight::dotransform(in, s);
   // final vector is the same size as straight-mapped one
   out.init(s);
   // shuffle the results
   for(int i=0; i<out.size(); i++)
      out(i) = s(lut(i));
   }

void map_permuted::doinverse(const libbase::matrix<double>& pin, libbase::matrix<double>& pout) const
   {
   // do the base (straight) mapping into a temporary space
   libbase::matrix<double> ptable;
   map_straight::doinverse(pin, ptable);
   // final matrix is the same size as straight-mapped one
   pout.init(ptable);
   // invert the shuffling
   assert(ptable.xsize() == lut.size());
   for(int i=0; i<pout.xsize(); i++)
      for(int j=0; j<pout.ysize(); j++)
         pout(lut(i),j) = ptable(i,j);
   }

// Description

std::string map_permuted::description() const
   {
   std::ostringstream sout;
   sout << "Interleaved Mapper";
   return sout.str();
   }

// Serialization Support

std::ostream& map_permuted::serialize(std::ostream& sout) const
   {
   map_straight::serialize(sout);
   return sout;
   }

std::istream& map_permuted::serialize(std::istream& sin)
   {
   map_straight::serialize(sin);
   return sin;
   }

}; // end namespace
