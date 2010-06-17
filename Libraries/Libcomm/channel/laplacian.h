#ifndef __laplacian_h
#define __laplacian_h

#include "config.h"
#include "channel.h"
#include "randgen.h"
#include "itfunc.h"
#include "serializer.h"
#include <cmath>

namespace libcomm {

/*!
 * \brief   Common Base for Additive Laplacian Noise Channel.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * \note The distribution has zero mean.
 */

template <class S, template <class > class C = libbase::vector>
class basic_laplacian : public channel<S, C> {
protected:
   // channel paremeters
   double lambda;
protected:
   // internal helper functions
   double f(const double x) const
      {
      return 1 / (2 * lambda) * exp(-fabs(x) / lambda);
      }
   double Finv(const double y) const
      {
      return (y < 0.5) ? lambda * log(2 * y) : -lambda * log(2 * (1 - y));
      }
public:
   // Description
   std::string description() const
      {
      return "Laplacian channel";
      }
};

/*!
 * \brief   General Additive Laplacian Noise Channel.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

template <class S, template <class > class C = libbase::vector>
class laplacian : public basic_laplacian<S, C> {
private:
   // Shorthand for class hierarchy
   typedef basic_laplacian<S, C> Base;
protected:
   // channel handle functions
   S corrupt(const S& s)
      {
      const S n = Base::Finv(Base::r.fval_closed());
      return s + n;
      }
   double pdf(const S& tx, const S& rx) const
      {
      const S n = rx - tx;
      return f(n);
      }
public:
   // Parameter handling
   void set_parameter(const double x)
      {
      assertalways(x >= 0);
      Base::lambda = x;
      }
   double get_parameter() const
      {
      return Base::lambda;
      }

   // Serialization Support
DECLARE_SERIALIZER(laplacian)
};

/*!
 * \brief   Signal-Space Additive Laplacian Noise Channel.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

template <>
class laplacian<sigspace, libbase::vector> : public basic_laplacian<sigspace> {
protected:
   // handle functions
   void compute_parameters(const double Eb, const double No);
   // channel handle functions
   sigspace corrupt(const sigspace& s);
   double pdf(const sigspace& tx, const sigspace& rx) const;
public:
   // Serialization Support
DECLARE_SERIALIZER(laplacian)
};

} // end namespace

#endif

