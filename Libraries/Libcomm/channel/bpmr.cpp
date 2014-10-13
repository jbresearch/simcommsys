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

#include "bpmr.h"
#include "itfunc.h"
#include <sstream>
#include <limits>
#include <exception>
#include <cmath>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show Markov state vector during transmission process
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*!
 * \brief Initialization
 *
 * Sets the channel with fixed values for Pd, Pi. This way, if the user
 * never calls set_parameter(), the values are valid.
 */
template <class real>
void bpmr<real>::init()
   {
   // channel parameters
   Pd = fixedPd;
   Pi = fixedPi;
   }

// Constructors / Destructors

/*!
 * \brief Principal constructor
 *
 * \sa init()
 */
template <class real>
bpmr<real>::bpmr(const bool varyPd, const bool varyPi) :
   varyPd(varyPd), varyPi(varyPi), fixedPd(0), fixedPi(0)
   {
   // channel update flags
   assert(varyPd || varyPi);
   // other initialization
   init();
   }

// Channel parameter handling

/*!
 * \brief Set channel parameter
 *
 * This function sets any of Pd, or Pi that are flagged to change. Any of
 * these parameters that are not flagged to change will instead be set to the
 * specified fixed value.
 *
 * \note We set fixed values every time to ensure that there is no leakage
 * between successive uses of this class. (i.e. once this function is called,
 * the class will be in a known determined state).
 */
template <class real>
void bpmr<real>::set_parameter(const double p)
   {
   set_pd(varyPd ? p : fixedPd);
   set_pi(varyPi ? p : fixedPi);
   libbase::trace << "DEBUG (bpmr): Pd = " << Pd << ", Pi = " << Pi
         << std::endl;
   }

/*!
 * \brief Get channel parameter
 *
 * This returns the value of the first of Pd, or Pi that are flagged to
 * change. If none of these are flagged to change, this constitutes an error
 * condition.
 */
template <class real>
double bpmr<real>::get_parameter() const
   {
   assert(varyPd || varyPi);
   if (varyPd)
      return Pd;
   // must be varyPi
   return Pi;
   }

// Channel functions

/*!
 * \copydoc channel::transmit()
 *
 * The channel model implemented is described by the following general case
 * state transition probabilities:
 *
 * Pr{Z_i = z+1 | Z_{i-1} = z} = P_i
 * Pr{Z_i = z-1 | Z_{i-1} = z} = P_d
 * Pr{Z_i = z   | Z_{i-1} = z} = 1-P_i-P_d
 *
 * with all other transitions having zero probability. Exceptions to the above
 * occur where z-1 or z+1 are not valid states:
 *
 * For z = Zmax:
 *   Pr{Z_i = z-1 | Z_{i-1} = z} = P_d
 *   Pr{Z_i = z   | Z_{i-1} = z} = 1-P_d
 * For z = Zmin:
 *   Pr{Z_i = z+1 | Z_{i-1} = z} = P_i
 *   Pr{Z_i = z   | Z_{i-1} = z} = 1-P_i
 *
 * It is assumed here (though not explicitly stated in Iyengar et al) that the
 * initial condition for the channel is Z_0 = 0 (where Z_1 refers to the first
 * input bit X_1 and output bit Y_1).
 * The channel output has the following correspondence with the input:
 *
 * Y_i = X_{i - Z_i}
 *
 * where it is further assumed that any X_i for an index 'i' outside the defined
 * range is equiprobable.
 *
 * \note We have to make sure that we don't corrupt the vector we're reading
 * from (in the case where tx and rx are the same vector); therefore,
 * the result is first created as a new vector and only copied over at
 * the end.
 *
 * \sa corrupt()
 */
template <class real>
void bpmr<real>::transmit(const array1b_t& tx, array1b_t& rx)
   {
   const int tau = tx.size();
   // Allocate and initialize Markov state sequence
   Z.init(tau);
   Z = 0;
   // determine state sequence
   int Zprev = 0;
   for (int i = 0; i < tau; i++)
      {
      const double p = this->r.fval_closed();
      // upper limit
      if (Zprev == Zmax)
         {
         if (p < Pd)
            Z(i) = Zprev - 1;
         else
            Z(i) = Zprev;
         }
      // lower limit
      else if (Zprev == Zmin)
         {
         if (p < Pi)
            Z(i) = Zprev + 1;
         else
            Z(i) = Zprev;
         }
      // general case
      else
         {
         if (p < Pi)
            Z(i) = Zprev + 1;
         else if (p < (Pi + Pd))
            Z(i) = Zprev - 1;
         else
            Z(i) = Zprev;
         }
      // copy back for next time
      Zprev = Z(i);
      }
#if DEBUG>=2
   libbase::trace << "DEBUG (bpmr): Z = " << Z << std::endl;
#endif
   // Initialize results vector
   array1b_t newrx(tau);
   // Compute the output vector (simulate the channel)
   for (int i = 0; i < tau; i++)
      {
      // determine the corresponding index into input vector
      const int j = i - Z(i);
      // valid index
      if (j >= 0 && j < tau)
         newrx(i) = tx(j);
      // invalid index -> equiprobable
      else
         newrx(i) = (this->r.fval_closed() < 0.5);
      }
   // copy results back
   rx = newrx;
   }

// description output

template <class real>
std::string bpmr<real>::description() const
   {
   std::ostringstream sout;
   sout << "BPMR channel (";
   // List set of valid states
   sout << "Z in [" << Zmin << ".." << Zmax << "], ";
   // List varying components
   if (varyPi)
      sout << "Pi=";
   if (varyPd)
      sout << "Pd=";
   sout << "p";
   // List non-varying components, with their value
   if (!varyPd)
      sout << ", Pd=" << fixedPd;
   if (!varyPi)
      sout << ", Pi=" << fixedPi;
   sout << ")";
#ifdef USE_CUDA
   sout << " [CUDA]";
#endif
   return sout.str();
   }

// object serialization - saving

template <class real>
std::ostream& bpmr<real>::serialize(std::ostream& sout) const
   {
   sout << "# Version" << std::endl;
   sout << 1 << std::endl;
   sout << "# Zmin" << std::endl;
   sout << Zmin << std::endl;
   sout << "# Zmax" << std::endl;
   sout << Zmax << std::endl;
   sout << "# Vary Pd?" << std::endl;
   sout << varyPd << std::endl;
   sout << "# Vary Pi?" << std::endl;
   sout << varyPi << std::endl;
   sout << "# Fixed Pd value" << std::endl;
   sout << fixedPd << std::endl;
   sout << "# Fixed Pi value" << std::endl;
   sout << fixedPi << std::endl;
   return sout;
   }

// object serialization - loading

/*!
 * \version 1 Initial version
 */
template <class real>
std::istream& bpmr<real>::serialize(std::istream& sin)
   {
   // get format version
   int version;
   sin >> libbase::eatcomments >> version >> libbase::verify;
   // read state range
   sin >> libbase::eatcomments >> Zmin >> libbase::verify;
   sin >> libbase::eatcomments >> Zmax >> libbase::verify;
   // read flags
   sin >> libbase::eatcomments >> varyPd >> libbase::verify;
   sin >> libbase::eatcomments >> varyPi >> libbase::verify;
   // read fixed Pd,Pi
   sin >> libbase::eatcomments >> fixedPd >> libbase::verify;
   sin >> libbase::eatcomments >> fixedPi >> libbase::verify;
   // sanity checks
   assertalways(Zmin <= 0);
   assertalways(Zmax > Zmin);
   // initialise the object and return
   init();
   return sin;
   }

} // end namespace

#include "mpgnu.h"
#include "logrealfast.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::mpgnu;
using libbase::logrealfast;

#ifdef USE_CUDA
#define REAL_TYPE_SEQ \
   (float)(double)
#else
#define REAL_TYPE_SEQ \
   (float)(double)(mpgnu)(logrealfast)
#endif

/* Serialization string: bpmr<real>
 * where:
 *      real = float | double | [mpgnu | logrealfast (CPU only)]
 */
#define INSTANTIATE(r, x, type) \
      template class bpmr<type>; \
      template <> \
      const serializer bpmr<type>::shelper( \
            "channel", \
            "bpmr<" BOOST_PP_STRINGIZE(type) ">", \
            bpmr<type>::create);

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, REAL_TYPE_SEQ)

} // end namespace
