/*!
 \file

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$
 */

#include "commsys_threshold.h"

namespace libcomm {

// Experiment parameter handling

template <class S, class R>
void commsys_threshold<S, R>::set_parameter(const double x)
   {
   parametric& m = dynamic_cast<parametric&> (*this->sys->getmodem());
   m.set_parameter(x);
   }

template <class S, class R>
double commsys_threshold<S, R>::get_parameter() const
   {
   const parametric& m =
         dynamic_cast<const parametric&> (*this->sys->getmodem());
   return m.get_parameter();
   }

// Serialization Support

template <class S, class R>
std::ostream& commsys_threshold<S, R>::serialize(std::ostream& sout) const
   {
   sout << this->sys->getchan()->get_parameter() << '\n';
   commsys_simulator<S, R>::serialize(sout);
   return sout;
   }

template <class S, class R>
std::istream& commsys_threshold<S, R>::serialize(std::istream& sin)
   {
   double x;
   sin >> libbase::eatcomments >> x;
   commsys_simulator<S, R>::serialize(sin);
   this->sys->getchan()->set_parameter(x);
   return sin;
   }

// Explicit Realizations

template class commsys_threshold<bool> ;
template <>
const libbase::serializer commsys_threshold<bool>::shelper("experiment",
      "commsys_threshold<bool>", commsys_threshold<bool>::create);

} // end namespace
