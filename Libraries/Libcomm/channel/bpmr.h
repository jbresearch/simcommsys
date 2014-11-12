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

#ifndef __bpmr_h
#define __bpmr_h

#include "config.h"
#include "bitfield.h"
#include "channel_insdel.h"
#include "serializer.h"
#include "matrix.h"
#include <cmath>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// NOTE: since this is a header, it may be included in other classes as well;
//       to avoid problems, the debug level is reset at the end of this file.
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*!
 * \brief   Bit-Patterned Media Recording channel.
 * \author  Johann Briffa
 *
 * This class implements the BPM recording channel using an extension of the
 * K-ary Markov state channel described in:
 * Iyengar, A.R., Siegel, P.H. and Wolf, J.K., "Write Channel Model for
 * Bit-Patterned Media Recording," IEEE Transactions on Magnetics, vol.47, no.1,
 * pp.35-45, Jan. 2011
 * URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5676449&isnumber=5676431
 *
 * The extension allows the Markov state to take negative values as well as
 * zero and positive values, effectively allowing deletion-before-insertion.
 *
 * \tparam real Floating-point type for internal computation
 *
 * \note Unlike the BSID and QIDS channels, this model has no concept of stream
 * operation, as at the receiving end, a full sector will always be retrieved.
 */

template <class real>
class bpmr : public channel_insdel<bool, real> {
public:
   /*! \name Type definitions */
   typedef libbase::matrix<real> array2r_t;
   typedef libbase::vector<real> array1r_t;
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<bool> array1b_t;
   typedef libbase::vector<double> array1d_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   typedef libbase::bitfield bitfield;
   // @}
private:
   /*! \name User-defined parameters */
   bool varyPd; //!< Flag to indicate that \f$ P_d \f$ should change with parameter
   bool varyPi; //!< Flag to indicate that \f$ P_i \f$ should change with parameter
   int Zmax; //!< Maximum value of Markov state (equal to K-1 in Iyengar paper)
   int Zmin; //!< Minimum value of Markov state (0 for Iyengar model, otherwise negative)
   double fixedPd; //!< Value to use when \f$ P_d \f$ does not change with parameter
   double fixedPi; //!< Value to use when \f$ P_i \f$ does not change with parameter
   // @}
public:
   /*! \name Metric computation */
   class metric_computer : public channel_insdel<bool, real>::metric_computer {
   public:
      /*! \name Channel-state and pre-computed parameters */
      real Pd; //!< Probability of deletion event
      real Pi; //!< Probability of insertion event
      int T; //!< block size in channel symbols
      int mT_min; //!< Assumed largest negative drift over a whole \c T channel-symbol block is \f$ m_T^{-} \f$
      int mT_max; //!< Assumed largest positive drift over a whole \c T channel-symbol block is \f$ m_T^{+} \f$
      // @}
      /*! \name Hardwired parameters */
      static const int arraysize = 128; //!< Size of stack-allocated arrays
      // @}
   private:
      real get_transmission_coefficient(int Z) const
         {
         assert(Z >= mT_min);
         assert(Z <= mT_max);
         if (mT_min == mT_max) // degenerate case with one state
            return 1;
         else if (Z == mT_min) // Z == mT_min
            return real(1) - Pi; // only insertion or transmission were possible
         else if (Z == mT_max) // Z == mT_max
            return real(1) - Pd; // only deletion or transmission were possible
         else // mT_min < Z < mT_max
            return real(1) - Pi - Pd; // general case: insertion, deletion, or transmission
         }
   public:
      /*! \name Internal functions */
      void precompute(double Pd, double Pi, int T, int mT_min, int mT_max);
      void init()
         {
#ifdef USE_CUDA
         // Initialize CUDA
         cuda::cudaInitialize(std::cerr);
#endif
         }
      // @}
#ifdef USE_CUDA
      /*! \name Device methods */
#ifdef __CUDACC__
      //! Receiver interface
      __device__
      real receive(const cuda::vector_reference<bool>& tx, const cuda::vector_reference<bool>& rx) const
         {
         // Compute sizes
         const int n = tx.size();
         const int mu = rx.size() - n;
         // Allocate space for results and call main receiver
         real ptable_data[arraysize];
         cuda_assertalways(arraysize >= mT_max - mT_min + 1);
         cuda::vector_reference<real> ptable(ptable_data, mT_max - mT_min + 1);
         receive(tx, rx, ptable);
         // return result
         return ptable(mu - mT_min);
         }
      //! Batch receiver interface
      __device__
      void receive(const cuda::vector_reference<bool>& tx, const cuda::vector_reference<bool>& rx,
            cuda::vector_reference<real>& ptable) const
         {
         using cuda::min;
         using cuda::max;
         using cuda::swap;
         // Compute sizes
         const int n = tx.size();
         const int rho = rx.size();
         // Set up single slice of lattice on the stack as a fixed size;
         // this avoids dynamic allocation (which would otherwise be necessary
         // as the size is non-const)
         /*
         cuda_assertalways(rho + 1 <= arraysize);
         real F[arraysize];
         */
         // set up variable to keep track of Fprev[j-1]
         real Fprev;
         // get access to slice of lattice in shared memory
         cuda::SharedMemory<real> smem;
         const int pitch = n + mT_max + 1;
         cuda_assertalways(rho + 1 <= pitch);
         __restrict__ real* F = smem.getPointer() + (threadIdx.x + threadIdx.y * blockDim.x) * pitch;
         // initialize for i=0 (first row of lattice)
         // Fthis[0] = 1;
         F[0] = 1;
         const int jmax = min(mT_max, rho);
         for (int j = 1; j <= jmax; j++)
            {
            // Fthis[j] = Fthis[j - 1] * Pval_i;
            F[j] = F[j - 1] * Pval_i;
            }
         // compute remaining rows, except last
         for (int i = 1; i < n; i++)
            {
            // keep Fprev[0]
            Fprev = F[0];
            // handle first column as a special case, if necessary
            if (i + mT_min <= 0)
               {
               // Fthis[0] = Fprev[0] * Pval_d;
               F[0] = Fprev * Pval_d;
               }
            // determine limits for remaining columns (after first)
            const int jmin = max(i + mT_min, 1);
            const int jmax = min(i + mT_max, rho);
            // keep Fprev[jmin - 1], if necessary
            if (jmin > 1)
               Fprev = F[jmin - 1];
            // remaining columns
            for (int j = jmin; j <= jmax; j++)
               {
               // transmission/substitution path
               const bool cmp = tx(i - 1) == rx(j - 1);
               // temp = Fprev[j - 1] * (cmp ? Pval_tc : Pval_te);
               real temp = Fprev * (cmp ? Pval_tc : Pval_te);
               // keep Fprev[j] for next time (to use as Fprev[j-1])
               Fprev = F[j];
               // deletion path (if previous row was within corridor)
               if (j < i + mT_max)
                  // temp += Fprev[j] * Pval_d;
                  temp += Fprev * Pval_d;
               // insertion path
               // temp += Fthis[j - 1] * Pval_i;
               temp += F[j - 1] * Pval_i;
               // store result
               // Fthis[j] = temp;
               F[j] = temp;
               }
            }
         // compute last row as a special case (no insertions)
         const int i = n;
         // keep Fprev[0]
         Fprev = F[0];
         // handle first column as a special case, if necessary
         if (i + mT_min <= 0)
            {
            // Fthis[0] = Fprev[0] * Pval_d;
            F[0] = Fprev * Pval_d;
            }
         // remaining columns
         for (int j = 1; j <= rho; j++)
            {
            // transmission/substitution path
            const bool cmp = tx(i - 1) == rx(j - 1);
            // temp = Fprev[j - 1] * (cmp ? Pval_tc : Pval_te);
            real temp = Fprev * (cmp ? Pval_tc : Pval_te);
            // keep Fprev[j] for next time (to use as Fprev[j-1])
            Fprev = F[j];
            // deletion path (if previous row was within corridor)
            if (j < i + mT_max)
               // temp += Fprev[j] * Pval_d;
               temp += Fprev * Pval_d;
            // store result
            // Fthis[j] = temp;
            F[j] = temp;
            }
         // copy results and return
         cuda_assertalways(ptable.size() == mT_max - mT_min + 1);
         for (int x = mT_min; x <= mT_max; x++)
            {
            // convert index
            const int j = x + n;
            if (j >= 0 && j <= rho)
               ptable(x - mT_min) = F[j];
            else
               ptable(x - mT_min) = 0;
            }
         }
#endif
      // @}
#endif
      /*! \name Host methods */
      //! Determine the amount of shared memory required per thread
      size_t receiver_sharedmem() const
         {
         return (T + mT_max + 1) * sizeof(real);
         }
      //! Receiver interface
      real receive(const bool& tx, const array1b_t& rx) const
         {
         failwith("Method not defined.");
         return 0;
         }
      //! Receiver interface
      real receive(const array1b_t& tx, const array1b_t& rx) const
         {
         // Compute sizes
         const int n = tx.size();
         const int mu = rx.size() - n;
         // Allocate space for results and call main receiver
         static array1r_t ptable;
         ptable.init(mT_max - mT_min + 1);
         receive(tx, rx, ptable);
         // return result
         return ptable(mu - mT_min);
         }
      //! Batch receiver interface
      void receive(const array1b_t& tx, const array1b_t& rx, array1r_t& ptable) const;
      // @}
      DECLARE_CLONABLE(metric_computer)
   };
   // @}
private:
   /*! \name Internal representation */
   metric_computer computer;
   double Pd; //!< Bit-deletion probability \f$ P_d \f$
   double Pi; //!< Bit-insertion probability \f$ P_i \f$
   int T; //!< Block size in channel symbols over which we want to synchronize
   array1i_t Z; //!< Markov state sequence; Z(i) = drift after 'i' channel uses
   // @}
private:
   /*! \name Internal functions */
   void init();
   void precompute()
      {
      if (T > 0)
         computer.precompute(Pd, Pi, T, Zmin, Zmax);
      }
   // @}
protected:
   // Channel function overrides
   bool corrupt(const bool& s)
      {
      failwith("Method not defined.");
      return s;
      }
   double pdf(const bool& tx, const bool& rx) const
      {
      failwith("Method not defined.");
      return 0;
      }
public:
   /*! \name Constructors / Destructors */
   bpmr(const bool varyPd = true, const bool varyPi = true);
   // @}

   /*! \name Channel parameter handling */
   void set_parameter(const double p);
   double get_parameter() const;
   // @}

   /*! \name Channel parameter setters */
   //! Set the bit-deletion probability
   void set_pd(const double Pd)
      {
      assert(Pd >= 0 && Pd <= 1);
      assert(Pi + Pd >= 0 && Pi + Pd <= 1);
      this->Pd = Pd;
      precompute();
      }
   //! Set the bit-insertion probability
   void set_pi(const double Pi)
      {
      assert(Pi >= 0 && Pi <= 1);
      assert(Pi + Pd >= 0 && Pi + Pd <= 1);
      this->Pi = Pi;
      precompute();
      }
   // @}

   /*! \name Channel parameter getters */
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

   // Insertion-deletion channel functions
   int get_drift(int t) const
      {
      // shorthand for length of last transmitted frame
      const int tau = Z.size();
      // sanity check
      assert(t >= 0);
      assert(t <= tau);
      // determine drift (fixed at 0 at start and end)
      int drift = 0;
      if (t > 0 && t < tau)
         drift = Z(t - 1);
      return drift;
      }
   /*!
    * \copydoc channel_insdel::set_pr()
    *
    * \note No need to keep this value as nothing depends on it
    */
   void set_pr(const double Pr)
      {
      assert(Pr > 0 && Pr < 1);
      }
   /*!
    * \copydoc channel_insdel::set_blocksize()
    *
    * \note We keep this value only because the sharedmem requirements depend
    * on it
    */
   void set_blocksize(int T)
      {
      if (this->T != T)
         {
         assert(T > 0);
         this->T = T;
         precompute();
         }
      }
   /*!
    * \copydoc channel_insdel::compute_limits()
    *
    * \note For this channel model, the limits are fixed
    * \note Drift at start/end of frame is also fixed at zero
    */
   void compute_limits(int tau, double Pr, int& lower, int& upper,
         const libbase::vector<double>& sof_pdf = libbase::vector<double>(),
         const int offset = 0) const
      {
      upper = Zmax;
      lower = Zmin;
      }

   // Channel functions
   void transmit(const array1b_t& tx, array1b_t& rx);
   using channel<bool>::receive;
   //! \note Used by bpmr::receive(tx, rx, ptable)
   double receive(const bool& tx, const array1b_t& rx) const
      {
      failwith("Method not defined.");
      return 0;
      }
   /*! \note Used by: direct_blockembedder, ssis, direct_blockmodem,
    * lut_modulator
    */
   void receive(const array1b_t& tx, const array1b_t& rx, array1vd_t& ptable) const
      {
      failwith("Method not defined.");
      }
   //! \note Used by dminner
   double receive(const array1b_t& tx, const array1b_t& rx) const
      {
      failwith("Method not defined.");
      return 0;
      }

   // Interface for CUDA
   const typename channel_insdel<bool, real>::metric_computer& get_computer() const
      {
      return computer;
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(bpmr)
};

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG
#endif

} // end namespace

#endif
