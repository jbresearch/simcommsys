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

#include "dids.h"
#include "itfunc.h"
#include <sstream>
#include <limits>
#include <exception>
#include <cmath>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show Markov state, error sequence, and tx/rx vectors during transmission process
// 3 - Show tx/rx vectors and posterior probabilities from receive process
// 4 - Show lattice rows as they are computed in receive process
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

// Internal functions

/*!
 * \brief Sets up pre-computed values
 *
 * This function computes all cached quantities used within actual channel
 * operations. Since these values depend on the channel conditions, this
 * function should be called any time a channel parameter is changed.
 */
template <class real>
void dids<real>::metric_computer::precompute(double Pd, double Pi, double Pr,
      double Pb, int T, int Zmin, int Zmax)
   {
   // block size
   this->T = T;
   // fba decoder parameters
   this->Zmin = Zmin;
   this->Zmax = Zmax;
   // channel parameters
   this->Pd = real(Pd);
   this->Pi = real(Pi);
   this->Pr = real(Pr);
   this->Pb = real(Pb);
   }

// Batch receiver interface
template <class real>
void dids<real>::metric_computer::receive(const array1b_t& tx,
      const array1b_t& rx, const int S0, const int delta0, const bool first,
      const bool last, array1r_t& ptable0, array1r_t& ptable1) const
   {
#if DEBUG>=3
   libbase::trace << "DEBUG (dids): starting receive..." << std::endl;
   libbase::trace << "DEBUG (dids): tx = " << tx << std::endl;
   libbase::trace << "DEBUG (dids): rx = " << rx << std::endl;
   libbase::trace << "DEBUG (dids): S0 = " << S0 << std::endl;
   libbase::trace << "DEBUG (dids): delta0 = " << delta0 << std::endl;
   libbase::trace << "DEBUG (dids): first = " << first << std::endl;
   libbase::trace << "DEBUG (dids): last = " << last << std::endl;
#endif
   using std::swap;
   using std::min;
   using std::max;
   /* Determine limits over this sequence
    * a) upper limit is the same as global limit
    * b) lower limit is bounded by ⌈T/2⌉ (one deletion per output bit)
    */
   const int mT_max = Zmax - S0;
   const int mT_min = last ? (Zmin - S0) : std::max(-int(std::ceil(T/2.0)), Zmin - S0);
#if DEBUG>=3
   libbase::trace << "DEBUG (dids): mT_max = " << mT_max << std::endl;
   libbase::trace << "DEBUG (dids): mT_min = " << mT_min << std::endl;
#endif
   // Compute sizes
   const int n = tx.size();
   const int rho = rx.size();
   // Set up three slices of lattice on the stack as a fixed size;
   // this avoids dynamic allocation (which would otherwise be necessary
   // as the size is non-const)
   assertalways(rho + 1 <= arraysize);
   real F[3][arraysize];
   real *F0 = &F[0][0];
   real *F1 = &F[1][0];
   real *F2 = &F[2][0];
#if DEBUG>=4
   // mark all lattice entries as uninitialized
   for (int i = 0; i < 3; i++)
   for (int j = 0; j <= rho; j++)
   F[i][j] = -1;
#endif
   // *** initialize first row of lattice (i = 0) [insertion only]
      {
      const int jmax = min(mT_max, rho);
      for (int j = 0; j <= jmax; j++)
         F0[j] = 0;
      if (delta0 == 0) // last bit of previous codeword not deleted
         {
         F0[0] = 1;
         if (first) // assume equiprobable prior value
            for (int j = 1; j <= jmax; j++)
               F0[j] = F0[j - 1] * real(0.5) * Pi;
         }
#if DEBUG>=4
      libbase::trace << "DEBUG (dids): F = " << std::endl;
      print_lattice_row(F0, rho);
#endif
      }
   // *** compute second row (i = 1)
      {
      int i = 1;
      assert(i <= n);
      // advance slices
      cycle_pointers(F0, F1, F2);
      // determine limits for remaining columns (after first)
      const int jmin = max(i + mT_min, 1);
      const int jmax = min(i + mT_max, rho);
      // overwrite up to three columns before corridor, as needed
      // [first is actually within corridor if (i + mT_min <= 0)]
      for (int j = jmin - 1; j >= 0 && j >= jmin - 3; j--)
         F0[j] = 0;
      // remaining columns
      if (delta0 == 0) // last bit of previous codeword not deleted
         {
         for (int j = jmin; j <= jmax; j++)
            {
            real temp = 0;
            // determine whether corresponding tx/rx bits are equal
            // [repeat last tx bit for virtual rows]
            const bool cmp = (tx(i - 1) == rx(j - 1));
            // transmission path
            temp += F1[j - 1] * (cmp ? real(1) - Pr : Pr)
                  * get_transmission_coefficient(j - i + S0);
            // insertion path (if previous node was within corridor)
            if (j - i > mT_min) // (j-1)-i >= mT_min
               temp += F0[j - 1] * (cmp ? real(1) - Pb : Pb) * Pi;
            // implicit free delete with no transmission at end of last codeword
            if (last && j - i < mT_max && j + S0 == n) // (j)-(i-1) <= mT_max
               temp += F1[j];
            // store result
            F0[j] = temp;
            }
         }
      else
         {
         assert(delta0 == 1);
         assert(jmin == 1);
         for (int j = jmin; j <= jmax; j++)
            {
            real temp = 0;
            // determine whether corresponding tx/rx bits are equal
            // [repeat last tx bit for virtual rows]
            const bool cmp = (tx(i - 1) == rx(j - 1));
            // deletion path across codeword boundary
            if (j == 1)
               temp += (cmp ? real(1) - Pr : Pr) * Pd;
            // insertion path (if previous node was within corridor)
            if (j - i > mT_min) // (j-1)-i >= mT_min
               temp += F0[j - 1] * (cmp ? real(1) - Pb : Pb) * Pi;
            // store result
            F0[j] = temp;
            }
         }
#if DEBUG>=4
      print_lattice_row(F0, rho);
#endif
      }
   // *** compute remaining rows (2 <= i <= n)
   // and -Zmin virtual rows if this is the last codeword
   const int imax = n + (last ? -Zmin : 0);
   for (int i = 2; i <= imax; i++)
      {
      // advance slices
      cycle_pointers(F0, F1, F2);
      // determine limits for remaining columns (after first)
      const int jmin = max(i + mT_min, 1);
      const int jmax = min(i + mT_max, rho);
      // overwrite up to three columns before corridor, as needed
      // [first is actually within corridor if (i + mT_min <= 0)]
      for (int j = jmin - 1; j >= 0 && j >= jmin - 3; j--)
         F0[j] = 0;
      // remaining columns
      for (int j = jmin; j <= jmax; j++)
         {
         real temp = 0;
         // determine whether corresponding tx/rx bits are equal
         // [repeat last tx bit for virtual rows]
         const bool cmp = (tx(std::min(i, n) - 1) == rx(j - 1));
         // transmission path
         temp += F1[j - 1] * (cmp ? real(1) - Pr : Pr)
               * get_transmission_coefficient(j - i + S0);
         // deletion path (if previous node was within corridor)
         if (j - i < mT_max) // (j-1)-(i-2) <= mT_max
            temp += F2[j - 1] * (cmp ? real(1) - Pr : Pr) * Pd;
         // insertion path (if previous node was within corridor)
         if (j - i > mT_min) // (j-1)-i >= mT_min
            temp += F0[j - 1] * (cmp ? real(1) - Pb : Pb) * Pi;
         // implicit free delete with no transmission at end of last codeword
         if (last && j - i < mT_max && j + S0 == n) // (j)-(i-1) <= mT_max
            temp += F1[j];
         // store result
         F0[j] = temp;
         }
#if DEBUG>=4
      print_lattice_row(F0, rho);
#endif
      }
   // *** copy results and return
   assertalways(ptable0.size() == Zmax - Zmin + 1);
   assertalways(ptable1.size() == Zmax - Zmin + 1);
   for (int x = Zmin; x <= Zmax; x++)
      {
      // convert index (x = j - n + S0)
      const int j = x + n - S0;
      // last tx bit not deleted
      if (j >= 0 && j <= rho)
         ptable0(x - Zmin) = F0[j];
      else
         ptable0(x - Zmin) = 0;
      // last tx bit deleted
      if (j >= 0 && j <= rho && j - n < mT_max) // (j)-(n-1) <= mT_max
         ptable1(x - Zmin) = F1[j];
      else
         ptable1(x - Zmin) = 0;
      }
#if DEBUG>=3
   libbase::trace << "DEBUG (dids): ptable0 = " << ptable0 << std::endl;
   libbase::trace << "DEBUG (dids): ptable1 = " << ptable1 << std::endl;
#endif
   }

/*!
 * \brief Initialization
 *
 * Sets the channel with fixed values for Pd, Pi, Pr, Pb. This way, if the
 * user never calls set_parameter(), the values are valid.
 */
template <class real>
void dids<real>::init()
   {
   // channel parameters
   Pd = fixedPd;
   Pi = fixedPi;
   Pr = fixedPr;
   Pb = fixedPb;
   // initialize metric computer
   computer.init();
   }

/*!
 * \brief Generate Markov state sequence
 *
 * The channel model implemented is described by the following general case
 * state transition probabilities:
 *
 * Pr{Z_i = z+1 | Z_{i-1} = z} = P_I
 * Pr{Z_i = z-1 | Z_{i-1} = z} = P_D
 * Pr{Z_i = z   | Z_{i-1} = z} = 1-P_I-P_D
 *
 * with all other transitions having zero probability. Exceptions to the above
 * occur where z-1 or z+1 are not valid states:
 *
 * For z = Zmax:
 *   Pr{Z_i = z-1 | Z_{i-1} = z} = P_D
 *   Pr{Z_i = z   | Z_{i-1} = z} = 1-P_D
 * For z = Zmin:
 *   Pr{Z_i = z+1 | Z_{i-1} = z} = P_I
 *   Pr{Z_i = z   | Z_{i-1} = z} = 1-P_I
 *
 * So far, this is the same as the BPMR channel. However, the DIDS channel
 * also has another exception, so that the same bit cannot be inserted more
 * than once. This is defined by:
 *
 * Pr{Z_i = z+1 | Z_{i-1} = z, Z_{i-2} = z-1} = 0
 *
 * It is assumed here that the initial condition for the channel is Z_0 = 0
 * (where Z_1 refers to the first input bit X_1 and output bit Y_1).
 */
template <class real>
void dids<real>::generate_state_sequence(const int tau)
   {
   // Allocate and initialize Markov state sequence
   Z.init(tau);
   Z = 0;
   // determine state sequence
   int Z1 = 0;
   int Z2 = 0;
   for (int i = 0; i < tau; i++)
      {
      const double p = this->r.fval_closed();
      // upper limit
      if (Z1 == Zmax)
         {
         if (p < Pd)
            Z(i) = Z1 - 1;
         else
            Z(i) = Z1;
         }
      else
      // lower limit
      if (Z1 == Zmin)
         {
         if (Z1 - Z2 < 1 && p < Pi)
            Z(i) = Z1 + 1;
         else
            Z(i) = Z1;
         }
      else // general case
         {
         if (p < Pd)
            Z(i) = Z1 - 1;
         else if (Z1 - Z2 < 1 && p < (Pi + Pd))
            Z(i) = Z1 + 1;
         else
            Z(i) = Z1;
         }
      // update for next round
      Z2 = Z1;
      Z1 = Z(i);
      }
#if DEBUG>=2
   libbase::trace << "DEBUG (dids): Z = " << Z << std::endl;
#endif
   }

/*!
 * \brief Generate substitution error sequence
 *
 * The channel model implemented is described by the following substitution
 * error probabilities:
 *
 * P_B for i' ∈ {i-L, ..., i+L} where Z_i - Z_{i-1} = 1 (insertion)
 *      or i' ∈ {i-L, ..., i+L-1} where Z_i - Z_{i-1} = -1 (deletion)
 * P_R otherwise
 */
template <class real>
libbase::vector<bool> dids<real>::generate_error_sequence()
   {
   // determine required length
   const int tau = Z.size();
   assert(tau > 0);
   // allocate and initialize burst sequence
   array1b_t B(tau);
   B = false;
   // determine burst sequence
   int Zprev = 0;
   for (int i = 0; i < tau; i++)
      {
      const int deltaZ = Z(i) - Zprev;
      if (deltaZ == 1) // inserted bits
         {
         const int imin = std::max(0, i - L);
         const int imax = std::min(tau - 1, i + L);
         B.segment(imin, imax - imin + 1) = true;
         }
      else if (deltaZ == -1) // deleted bits
         {
         const int imin = std::max(0, i - L);
         const int imax = std::min(tau - 1, i + L - 1);
         B.segment(imin, imax - imin + 1) = true;
         }
      else
         assert(deltaZ == 0);
      // update for next round
      Zprev = Z(i);
      }
   // allocate error sequence
   array1b_t E(tau);
   // determine error sequence
   for (int i = 0; i < tau; i++)
      {
      const double p = this->r.fval_closed();
      if (B(i))
         E(i) = (p < Pb); // burst-error bits
      else
         E(i) = (p < Pr); // random-error bits
      }
#if DEBUG>=2
   libbase::trace << "DEBUG (dids): B = " << B << std::endl;
   libbase::trace << "DEBUG (dids): E = " << E << std::endl;
#endif
   return E;
   }

// Constructors / Destructors

/*!
 * \brief Principal constructor
 *
 * \sa init()
 */
template <class real>
dids<real>::dids(const bool varyPd, const bool varyPi, const bool varyPr,
      const bool varyPb) :
      varyPd(varyPd), varyPi(varyPi), varyPr(varyPr), varyPb(varyPb), Zmax(1), Zmin(
            0), L(0), fixedPd(0), fixedPi(0), fixedPr(0), fixedPb(0)
   {
   // channel update flags
   assert(varyPd || varyPi || varyPr || varyPb);
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
void dids<real>::set_parameter(const double p)
   {
   set_pd(varyPd ? p : fixedPd);
   set_pi(varyPi ? p : fixedPi);
   set_ps(varyPr ? p : fixedPr);
   set_pb(varyPb ? p : fixedPb);
   libbase::trace << "DEBUG (dids): Pd = " << Pd << ", Pi = " << Pi << ", Pr = "
         << Pr << ", Pb = " << Pb << std::endl;
   }

/*!
 * \brief Get channel parameter
 *
 * This returns the value of the first of Pd, or Pi that are flagged to
 * change. If none of these are flagged to change, this constitutes an error
 * condition.
 */
template <class real>
double dids<real>::get_parameter() const
   {
   assert(varyPd || varyPi || varyPr || varyPb);
   if (varyPd)
      return Pd;
   if (varyPi)
      return Pi;
   if (varyPr)
      return Pr;
   // must be varyPsi
   return Pb;
   }

// Channel functions

/*!
 * \copydoc channel::transmit()
 *
 * The channel output has the following correspondence with the input:
 *
 * Y_i = X_{i - Z_i}
 *
 * where any input X_i for an index 'i' before the defined range is
 * equiprobable (representing the unknown prior magnetic state of the first
 * island, while input X_i for index 'i' after the defined range is the same
 * as the last valid input X_i.
 *
 * \note We have to make sure that we don't corrupt the vector we're reading
 * from (in the case where tx and rx are the same vector); therefore,
 * the result is first created as a new vector and only copied over at
 * the end.
 *
 * \sa corrupt()
 */
template <class real>
void dids<real>::transmit(const array1b_t& tx, array1b_t& rx)
   {
   const int tau = tx.size();
   // Generate Markov state sequence
   generate_state_sequence(tau);
   // Generate substitution error sequence
   const array1b_t E = generate_error_sequence();
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
      // early index -> equiprobable
      else if (j < 0)
         newrx(i) = (this->r.fval_closed() < 0.5);
      // late index -> repeat last valid input
      else
         newrx(i) = tx(tau - 1);
      }
   // Apply substitution errors, as applicable
   for (int i = 0; i < tau; i++)
      {
      if (E(i))
         newrx(i) = !newrx(i);
      }
   // copy results back
   rx = newrx;
#if DEBUG>=2
   libbase::trace << "DEBUG (dids): tx = " << tx << std::endl;
   libbase::trace << "DEBUG (dids): rx = " << rx << std::endl;
#endif
   }

// description output

template <class real>
std::string dids<real>::description() const
   {
   std::ostringstream sout;
   sout << "DIDS channel (";
   // Specify burst length
   sout << "L=" << L << ", ";
   // List varying components
   if (varyPi)
      sout << "Pi=";
   if (varyPd)
      sout << "Pd=";
   if (varyPr)
      sout << "Pr=";
   if (varyPb)
      sout << "Pb=";
   sout << "p";
   // List non-varying components, with their value
   if (!varyPd)
      sout << ", Pd=" << fixedPd;
   if (!varyPi)
      sout << ", Pi=" << fixedPi;
   if (!varyPr)
      sout << ", Pr=" << fixedPr;
   if (!varyPb)
      sout << ", Pb=" << fixedPb;
   sout << ")";
   return sout.str();
   }

// object serialization - saving

template <class real>
std::ostream& dids<real>::serialize(std::ostream& sout) const
   {
   sout << "# Version" << std::endl;
   sout << 1 << std::endl;
   sout << "# L" << std::endl;
   sout << L << std::endl;
   sout << "# Vary Pd?" << std::endl;
   sout << varyPd << std::endl;
   sout << "# Vary Pi?" << std::endl;
   sout << varyPi << std::endl;
   sout << "# Vary Pr?" << std::endl;
   sout << varyPr << std::endl;
   sout << "# Vary Pb?" << std::endl;
   sout << varyPb << std::endl;
   sout << "# Fixed Pd value" << std::endl;
   sout << fixedPd << std::endl;
   sout << "# Fixed Pi value" << std::endl;
   sout << fixedPi << std::endl;
   sout << "# Fixed Pr value" << std::endl;
   sout << fixedPr << std::endl;
   sout << "# Fixed Pb value" << std::endl;
   sout << fixedPb << std::endl;
   return sout;
   }

// object serialization - loading

/*!
 * \version 1 Initial version
 */
template <class real>
std::istream& dids<real>::serialize(std::istream& sin)
   {
   // get format version
   int version;
   sin >> libbase::eatcomments >> version >> libbase::verify;
   // read burst length
   sin >> libbase::eatcomments >> L >> libbase::verify;
   // read flags
   sin >> libbase::eatcomments >> varyPd >> libbase::verify;
   sin >> libbase::eatcomments >> varyPi >> libbase::verify;
   sin >> libbase::eatcomments >> varyPr >> libbase::verify;
   sin >> libbase::eatcomments >> varyPb >> libbase::verify;
   // read fixed error rates
   sin >> libbase::eatcomments >> fixedPd >> libbase::verify;
   sin >> libbase::eatcomments >> fixedPi >> libbase::verify;
   sin >> libbase::eatcomments >> fixedPr >> libbase::verify;
   sin >> libbase::eatcomments >> fixedPb >> libbase::verify;
   // sanity checks
   assertalways(L >= 0);
   // fixed values
   Zmin = -1;
   Zmax = 1;
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

#define REAL_TYPE_SEQ \
   (float)(double)(mpgnu)(logrealfast)

/* Serialization string: dids<real>
 * where:
 *      real = float | double | [mpgnu | logrealfast (CPU only)]
 */
#define INSTANTIATE(r, x, type) \
      template class dids<type>; \
      template <> \
      const serializer dids<type>::shelper( \
            "channel", \
            "dids<" BOOST_PP_STRINGIZE(type) ">", \
            dids<type>::create);

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, REAL_TYPE_SEQ)

} // end namespace
