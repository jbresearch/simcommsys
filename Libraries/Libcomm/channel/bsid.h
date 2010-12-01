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
 * 
 * \section svn Version Control
 * - $Id$
 */

#ifndef __bsid_h
#define __bsid_h

#include "config.h"
#include "bitfield.h"
#include "channel.h"
#include "itfunc.h"
#include "serializer.h"
#include "matrix.h"
#include "cuda-all.h"
#include "cuda/bitfield.h"
#include <cmath>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show tx and rx vectors when computing RecvPr
// NOTE: since this is a header, it may be included in other classes as well;
//       to avoid problems, the debug level is reset at the end of this file.
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*!
 * \brief   Binary substitution/insertion/deletion channel.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

class bsid : public channel<bool> {
private:
   // Shorthand for class hierarchy
   typedef channel<bool> Base;
public:
   /*! \name Type definitions */
   typedef float real;
   typedef libbase::matrix<real> array2r_t;
   typedef libbase::vector<bool> array1b_t;
   typedef libbase::vector<double> array1d_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   typedef libbase::bitfield bitfield;
   // @}
private:
   /*! \name User-defined parameters */
   bool biased; //!< Flag to indicate old-style bias against single-deletion
   bool varyPs; //!< Flag to indicate that \f$ P_s \f$ should change with parameter
   bool varyPd; //!< Flag to indicate that \f$ P_d \f$ should change with parameter
   bool varyPi; //!< Flag to indicate that \f$ P_i \f$ should change with parameter
   int Icap; //!< Maximum usable value of I (0 indicates no cap is placed)
   // @}
public:
   /*! \name Metric computation */
   class metric_computer {
   public:
      /*! \name Channel-state and pre-computed parameters */
      int N; //!< Block size in bits over which we want to synchronize
      int I; //!< Assumed limit for insertions between two time-steps
      int xmax; //!< Assumed maximum drift over a whole \c N -bit block
      real Rval; //!< Receiver coefficient value for mu = -1
#ifdef USE_CUDA
      cuda::matrix_auto<real> Rtable; //!< Receiver coefficient set for mu >= 0
#else
      array2r_t Rtable; //!< Receiver coefficient set for mu >= 0
#endif
      // @}
   public:
      /*! \name FBA decoder parameter computation */
      static int compute_I(int tau, double p, int Icap);
      static int compute_xmax(int tau, double p, int I);
      static real compute_Rtable_entry(bool err, int mu, double Ps, double Pd,
            double Pi);
      static void compute_Rtable(array2r_t& Rtable, int I, double Ps,
            double Pd, double Pi);
      // @}
      /*! \name Internal functions */
      void precompute(double Ps, double Pd, double Pi, int Icap, bool biased);
      void init();
      // @}
#ifdef USE_CUDA
      /*! \name Device methods */
#ifdef __CUDACC__
      __device__
      real receive(const cuda::bitfield& tx, const cuda::vector_reference<bool>& rx) const
         {
         using cuda::min;
         using cuda::max;
         using cuda::swap;
         // Compute sizes
         const int n = tx.size();
         const int mu = rx.size() - n;
         cuda_assert(n <= N);
         cuda_assert(labs(mu) <= xmax);
         // Set up two slices of forward matrix, and associated pointers
         // Arrays are allocated on the stack as a fixed size; this avoids dynamic
         // allocation (which would otherwise be necessary as the size is non-const)
         const int arraysize = 2 * 15 + 1;
         cuda_assert(2 * xmax + 1 <= arraysize);
         real F0[arraysize];
         real F1[arraysize];
         real *Fthis = F1;
         real *Fprev = F0;
         // for prior list, reset all elements to zero
         for (int x = 0; x < 2 * xmax + 1; x++)
            {
            Fprev[x] = 0;
            }
         // we also know x[0] = 0; ie. drift before transmitting bit t0 is zero.
         Fprev[xmax + 0] = 1;
         // compute remaining matrix values
         for (int j = 1; j < n; ++j)
            {
            // for this list, reset all elements to zero
            for (int x = 0; x < 2 * xmax + 1; x++)
               {
               Fthis[x] = 0;
               }
            // event must fit the received sequence:
            // 1. j-1+a >= 0
            // 2. j-1+y < rx.size()
            // limits on insertions and deletions must be respected:
            // 3. y-a <= I
            // 4. y-a >= -1
            // note: a and y are offset by xmax
            const int ymin = max(0, xmax - j);
            const int ymax = min(2 * xmax, xmax + rx.size() - j);
            for (int y = ymin; y <= ymax; ++y)
               {
               real result = 0;
               const int amin = max(max(0, xmax + 1 - j), y - I);
               const int amax = min(2 * xmax, y + 1);
               // check if the last element is a pure deletion
               int amax_act = amax;
               if (y - amax < 0)
                  {
                  result += Fprev[amax] * Rval;
                  amax_act--;
                  }
               // elements requiring comparison of tx and rx bits
               for (int a = amin; a <= amax_act; ++a)
                  {
                  // received subsequence has
                  // start:  j-1+a
                  // length: y-a+1
                  // therefore last element is: start+length-1 = j+y-1
                  const bool cmp = tx(j - 1) != rx(j + (y - xmax) - 1);
                  result += Fprev[a] * Rtable(cmp, y - a);
                  }
               Fthis[y] = result;
               }
            // swap 'this' and 'prior' lists
            swap(Fthis, Fprev);
            }
         // Compute forward metric for known drift, and return
         real result = 0;
         // event must fit the received sequence:
         // 1. n-1+a >= 0
         // 2. n-1+mu < rx.size() [automatically satisfied by definition of mu]
         // limits on insertions and deletions must be respected:
         // 3. mu-a <= I
         // 4. mu-a >= -1
         // note: muoff and a are offset by xmax
         const int muoff = mu + xmax;
         const int amin = max(max(0, muoff - I), xmax + 1 - n);
         const int amax = min(2 * xmax, muoff + 1);
         // check if the last element is a pure deletion
         int amax_act = amax;
         if (muoff - amax < 0)
            {
            result += Fprev[amax] * Rval;
            amax_act--;
            }
         // elements requiring comparison of tx and rx bits
         for (int a = amin; a <= amax_act; ++a)
            {
            // received subsequence has
            // start:  n-1+a
            // length: mu-a+1
            // therefore last element is: start+length-1 = n+mu-1
            const bool cmp = tx(n - 1) != rx(n + mu - 1);
            result += Fprev[a] * Rtable(cmp, muoff - a);
            }
         // clean up and return
         return result;
         }
#endif
      // @}
      /*! \name Kernel starters */
      real receive(const bitfield& tx, const array1b_t& rx) const;
      // @}
#else
      /*! \name Host methods */
      real receive(const bitfield& tx, const array1b_t& rx) const;
      // @}
#endif
   };
   // @}
private:
   /*! \name Metric computation */
   metric_computer computer;
   double Ps; //!< Bit-substitution probability \f$ P_s \f$
   double Pd; //!< Bit-deletion probability \f$ P_d \f$
   double Pi; //!< Bit-insertion probability \f$ P_i \f$
   // @}
private:
   /*! \name Internal functions */
   void init();
   // @}
protected:
   // Channel function overrides
   bool corrupt(const bool& s);
   double pdf(const bool& tx, const bool& rx) const
      {
      return (tx != rx) ? Ps : 1 - Ps;
      }
public:
   /*! \name Constructors / Destructors */
   bsid(const bool varyPs = true, const bool varyPd = true, const bool varyPi =
         true, const bool biased = false);
   // @}

   /*! \name FBA decoder parameter computation */
   /*!
    * \copydoc bsid::metric_computer::compute_I()
    *
    * \note Provided for use by clients; depends on object parameters
    */
   int compute_I(int tau)
      {
      return metric_computer::compute_I(tau, Pi, Icap);
      }
   /*!
    * \copydoc bsid::metric_computer::compute_xmax()
    *
    * \note Provided for use by clients; depends on object parameters
    */
   int compute_xmax(int tau)
      {
      const int I = metric_computer::compute_I(tau, Pi, Icap);
      return metric_computer::compute_xmax(tau, Pi, I);
      }
   // @}

   /*! \name Channel parameter handling */
   void set_parameter(const double p);
   double get_parameter() const;
   // @}

   /*! \name Channel parameter setters */
   //! Set the bit-substitution probability
   void set_ps(const double Ps)
      {
      assert(Ps >= 0 && Ps <= 0.5);
      this->Ps = Ps;
      }
   //! Set the bit-deletion probability
   void set_pd(const double Pd)
      {
      assert(Pd >= 0 && Pd <= 1);
      assert(Pi + Pd >= 0 && Pi + Pd <= 1);
      this->Pd = Pd;
      computer.precompute(Ps, Pd, Pi, Icap, biased);
      }
   //! Set the bit-insertion probability
   void set_pi(const double Pi)
      {
      assert(Pi >= 0 && Pi <= 1);
      assert(Pi + Pd >= 0 && Pi + Pd <= 1);
      this->Pi = Pi;
      computer.precompute(Ps, Pd, Pi, Icap, biased);
      }
   //! Set the block size
   void set_blocksize(int N)
      {
      if (N != computer.N)
         {
         assert(N > 0);
         computer.N = N;
         computer.precompute(Ps, Pd, Pi, Icap, biased);
         }
      }
   // @}

   /*! \name Channel parameter getters */
   //! Get the current bit-substitution probability
   double get_ps() const
      {
      return Ps;
      }
   //! Get the current bit-deletion probability
   double get_pd() const
      {
      return Pd;
      }
   //! Get the current bit-insertion probability
   double get_pi() const
      {
      return Pi;
      }
   // @}

   // Channel functions
   void transmit(const array1b_t& tx, array1b_t& rx);
   using Base::receive;
   void
   receive(const array1b_t& tx, const array1b_t& rx, array1vd_t& ptable) const;
   double receive(const array1b_t& tx, const array1b_t& rx) const
      {
#if DEBUG>=2
      libbase::trace << "DEBUG (bsid): Computing RecvPr for" << std::endl;
      libbase::trace << "tx = " << tx;
      libbase::trace << "rx = " << rx;
#endif
      const real result = computer.receive(bitfield(tx), rx);
#if DEBUG>=2
      libbase::trace << "RecvPr = " << result << std::endl;
#endif
      return result;
      }
   double receive(const bool& tx, const array1b_t& rx) const
      {
      // Compute sizes
      const int mu = rx.size() - 1;
      // If this was not a deletion, return result from table
      if (mu >= 0)
         return computer.compute_Rtable_entry(tx != rx(mu), mu, Ps, Pd, Pi);
      // If this was a deletion, it's a fixed value
      return computer.Rval;
      }

   // Interface for CUDA
   const metric_computer& get_computer() const
      {
      return computer;
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(bsid)
};

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG
#endif

} // end namespace

#endif
