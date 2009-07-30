/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "uniform_lut.h"
#include <sstream>

namespace libcomm {

// initialisation

template <class real>
void uniform_lut<real>::init(const int tau, const int m)
   {
   uniform_lut<real>::tau = tau;
   uniform_lut<real>::m = m;
   this->lut.init(tau);
   }

// intra-frame functions

template <class real>
void uniform_lut<real>::seedfrom(libbase::random& r)
   {
   this->r.seed(r.ival());
   advance();
   }

template <class real>
void uniform_lut<real>::advance()
   {
   // create array to hold 'used' status of possible lut values
   libbase::vector<bool> used(tau - m);
   used = false;
   // fill in lut
   int t;
   for (t = 0; t < tau - m; t++)
      {
      int tdash;
      do
         {
         tdash = r.ival(tau - m);
         } while (used(tdash));
      used(tdash) = true;
      this->lut(t) = tdash;
      }
   for (t = tau - m; t < tau; t++)
      this->lut(t) = fsm::tail;
   }

// description output

template <class real>
std::string uniform_lut<real>::description() const
   {
   std::ostringstream sout;
   sout << "Uniform Interleaver";
   if (m > 0)
      sout << " (Forced tail length " << m << ")";
   return sout.str();
   }

// object serialization - saving

template <class real>
std::ostream& uniform_lut<real>::serialize(std::ostream& sout) const
   {
   sout << this->lut.size() << "\n";
   sout << m << "\n";
   return sout;
   }

// object serialization - loading

template <class real>
std::istream& uniform_lut<real>::serialize(std::istream& sin)
   {
   int tau, m;
   sin >> libbase::eatcomments >> tau >> m;
   init(tau, m);
   return sin;
   }

// Explicit instantiations

template class uniform_lut<float> ;
template <>
const libbase::serializer uniform_lut<float>::shelper("interleaver",
      "uniform_lut<float>", uniform_lut<float>::create);

template class uniform_lut<double> ;
template <>
const libbase::serializer uniform_lut<double>::shelper("interleaver",
      "uniform_lut<double>", uniform_lut<double>::create);

template class uniform_lut<libbase::logrealfast> ;
template <>
const libbase::serializer uniform_lut<libbase::logrealfast>::shelper(
      "interleaver", "uniform_lut<logrealfast>", uniform_lut<
            libbase::logrealfast>::create);

} // end namespace
