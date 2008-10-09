/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "map_interleaved.h"
#include <stdlib.h>
#include <sstream>

namespace libcomm {

// Serialization Support

const libbase::serializer map_interleaved::shelper("mapper", "map_interleaved", map_interleaved::create);

// Interface with mapper

void map_interleaved::advance() const
   {
   lut.init(tau,r);
   }

void map_interleaved::dotransform(const libbase::vector<int>& in, libbase::vector<int>& out) const
   {
   // do the base (straight) mapping into a temporary space
   libbase::vector<int> s;
   map_straight::dotransform(in, s);
   // final vector is the same size as straight-mapped one
   out.init(s);
   // shuffle the results
   assert(out.size() == lut.size());
   for(int i=0; i<out.size(); i++)
      out(i) = s(lut(i));
   }

void map_interleaved::doinverse(const libbase::matrix<double>& pin, libbase::matrix<double>& pout) const
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

std::string map_interleaved::description() const
   {
   std::ostringstream sout;
   sout << "Interleaved Mapper";
   return sout.str();
   }

// Serialization Support

std::ostream& map_interleaved::serialize(std::ostream& sout) const
   {
   map_straight::serialize(sout);
   return sout;
   }

std::istream& map_interleaved::serialize(std::istream& sin)
   {
   map_straight::serialize(sin);
   return sin;
   }

}; // end namespace
