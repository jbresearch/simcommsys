/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

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
      const S n = S(Base::Finv(Base::r.fval_closed()));
      return s + n;
      }
   double pdf(const S& tx, const S& rx) const
      {
      const S n = rx - tx;
      return this->f(n);
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
 */

template <template <class > class C>
class laplacian<sigspace, C> : public basic_laplacian<sigspace, C> {
private:
   // Shorthand for class hierarchy
   typedef basic_laplacian<sigspace, C> Base;
protected:
   // handle functions
   void compute_parameters(const double Eb, const double No)
      {
      const double sigma = sqrt(Eb * No);
      Base::lambda = sigma / sqrt(double(2));
      }
   // channel handle functions
   sigspace corrupt(const sigspace& s)
      {
      const double x = Base::Finv(Base::r.fval_closed());
      const double y = Base::Finv(Base::r.fval_closed());
      return s + sigspace(x, y);
      }
   double pdf(const sigspace& tx, const sigspace& rx) const
      {
      sigspace n = rx - tx;
      return Base::f(n.i()) * Base::f(n.q());
      }
public:
   // Serialization Support
DECLARE_SERIALIZER(laplacian)
};

} // end namespace

#endif

