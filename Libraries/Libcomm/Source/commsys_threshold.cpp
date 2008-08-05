/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "commsys_threshold.h"
#include "dminner.h"

namespace libcomm {

// Experiment parameter handling

template <class S, class R>
void commsys_threshold<S,R>::set_parameter(const double x)
   {
   parametric& m = dynamic_cast<parametric&>(*this->modem);
   m.set_parameter(x);
   }

template <class S, class R>
double commsys_threshold<S,R>::get_parameter() const
   {
   const parametric& m = dynamic_cast<const parametric&>(*this->modem);
   return m.get_parameter();
   }

// Serialization Support

template <class S, class R>
std::ostream& commsys_threshold<S,R>::serialize(std::ostream& sout) const
   {
   sout << this->chan->get_parameter() << '\n';
   commsys<S,R>::serialize(sout);
   return sout;
   }

template <class S, class R>
std::istream& commsys_threshold<S,R>::serialize(std::istream& sin)
   {
   double x;
   sin >> x;
   commsys<S,R>::serialize(sin);
   this->chan->set_parameter(x);
   return sin;
   }

// Explicit Realizations

template class commsys_threshold<bool>;
template <>
const libbase::serializer commsys_threshold<bool>::shelper("experiment", "commsys_threshold<bool>", commsys_threshold<bool>::create);

}; // end namespace
