/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "mapper.h"

namespace libcomm {

// Setup functions

void mapper::set_parameters(const int N, const int M, const int S)
   {
   mapper::N = N;
   mapper::M = M;
   mapper::S = S;
   setup();
   }

// Vector mapper operations

void mapper::transform(const libbase::vector<int>& in, libbase::vector<int>& out) const
   {
   advance();
   advanced = true;
   dotransform(in, out);
   }

void mapper::inverse(const libbase::matrix<double>& pin, libbase::matrix<double>& pout) const
   {
   if(!advanced)
      advance();
   doinverse(pin, pout);
   advanced = false;
   }

}; // end namespace
