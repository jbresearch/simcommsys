/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "laplacian.h"

namespace libcomm {

// *** general template ***

// object serialization

template <class S, template <class > class C>
std::ostream& laplacian<S, C>::serialize(std::ostream& sout) const
   {
   return sout;
   }

template <class S, template <class > class C>
std::istream& laplacian<S, C>::serialize(std::istream& sin)
   {
   return sin;
   }

// Explicit instantiations

using libbase::serializer;
using libbase::vector;
using libbase::matrix;

template class laplacian<int> ;
template <>
const serializer laplacian<int>::shelper("channel", "laplacian<int>",
      laplacian<int>::create);

template class laplacian<float> ;
template <>
const serializer laplacian<float>::shelper("channel", "laplacian<float>",
      laplacian<float>::create);

template class laplacian<double> ;
template <>
const serializer laplacian<double>::shelper("channel", "laplacian<double>",
      laplacian<double>::create);

template class laplacian<int, matrix> ;
template <>
const serializer laplacian<int, matrix>::shelper("channel",
      "laplacian<int,matrix>", laplacian<int, matrix>::create);

template class laplacian<float, matrix> ;
template <>
const serializer laplacian<float, matrix>::shelper("channel",
      "laplacian<float,matrix>", laplacian<float, matrix>::create);

template class laplacian<double, matrix> ;
template <>
const serializer laplacian<double, matrix>::shelper("channel",
      "laplacian<double,matrix>", laplacian<double, matrix>::create);

// *** sigspace specialization ***

// handle functions

void laplacian<sigspace>::compute_parameters(const double Eb, const double No)
   {
   const double sigma = sqrt(Eb * No);
   lambda = sigma / sqrt(double(2));
   }

// channel handle functions

sigspace laplacian<sigspace>::corrupt(const sigspace& s)
   {
   const double x = Finv(r.fval_closed());
   const double y = Finv(r.fval_closed());
   return s + sigspace(x, y);
   }

double laplacian<sigspace>::pdf(const sigspace& tx, const sigspace& rx) const
   {
   sigspace n = rx - tx;
   return f(n.i()) * f(n.q());
   }

// object serialization

std::ostream& laplacian<sigspace>::serialize(std::ostream& sout) const
   {
   return sout;
   }

std::istream& laplacian<sigspace>::serialize(std::istream& sin)
   {
   return sin;
   }

// Explicit instantiations

const libbase::serializer laplacian<sigspace>::shelper("channel",
      "laplacian<sigspace>", laplacian<sigspace>::create);

} // end namespace
