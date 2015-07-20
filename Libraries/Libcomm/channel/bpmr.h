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
   bool varyPs; //!< Flag to indicate that \f$ P_s \f$ should change with parameter
   bool varyPsi; //!< Flag to indicate that \f$ P_{si} \f$ should change with parameter
   int Zmax; //!< Maximum value of Markov state (equal to K-1 in Iyengar paper)
   int Zmin; //!< Minimum value of Markov state (0 for Iyengar model, otherwise negative)
   double fixedPd; //!< Value to use when \f$ P_d \f$ does not change with parameter
   double fixedPi; //!< Value to use when \f$ P_i \f$ does not change with parameter
   double fixedPs; //!< Value to use when \f$ P_s \f$ does not change with parameter
   double fixedPsi; //!< Value to use when \f$ P_{si} \f$ does not change with parameter
   // @}
public:
   /*! \name Metric computation */
   class metric_computer : public channel_insdel<bool, real>::metric_computer {
   public:
      /*! \name Channel-state and pre-computed parameters */
      real Pd; //!< Probability of deletion event
      real Pi; //!< Probability of insertion event
      real Ps; //!< Probability of substitution error on non-inserted bits
      real Psi; //!< Probability of substitution error on inserted bits
      int T; //!< block size in channel symbols
      int Zmin; //!< Largest negative drift possible
      int Zmax; //!< Largest positive drift possible
      // @}
      /*! \name Hardwired parameters */
      static const int arraysize = 128; //!< Size of stack-allocated arrays
      // @}
   private:
      real get_transmission_coefficient(int Z) const
         {
         cuda_assert(Z >= Zmin);
         cuda_assert(Z <= Zmax);
         if (Zmin == Zmax) // degenerate case with one state
            return 1;
         else if (Z == Zmin) // Z == mT_min
            return real(1) - Pi; // only insertion or transmission were possible
         else if (Z == Zmax) // Z == mT_max
            return real(1) - Pd; // only deletion or transmission were possible
         else // mT_min < Z < mT_max
            return real(1) - Pi - Pd; // general case: insertion, deletion, or transmission
         }
      static void cycle_pointers(real* &F0, real* &F1, real* &F2)
         {
         real *Ft = F2;
         F2 = F1;
         F1 = F0;
         F0 = Ft;
         }
      static void print_lattice_row(const real *F0, const int rho)
         {
         for (int j = 0; j <= rho; j++)
            libbase::trace << F0[j] << '\t';
         libbase::trace << std::endl;
         }
   public:
      /*! \name Internal functions */
      void precompute(double Pd, double Pi, double Ps, double Psi, int T,
            int Zmin, int Zmax);
      // @}
      /*! \name Host methods */
      //! Determine the amount of shared memory required per thread
      size_t receiver_sharedmem() const
         {
         return 3 * (T + Zmax + 1) * sizeof(real);
         }
      //! Batch receiver interface - indefinite state space
      void receive(const array1b_t& tx, const array1b_t& rx,
            array1r_t& ptable) const
         {
         failwith("Method not supported.");
         }
      //! Batch receiver interface - fixed state space
      void receive(const array1b_t& tx, const array1b_t& rx, const int S0,
            const int delta0, const bool first, const bool last,
            array1r_t& ptable0, array1r_t& ptable1) const;
      // @}
      DECLARE_CLONABLE(metric_computer)
   };
   // @}
private:
   /*! \name Internal representation */
   metric_computer computer;
   double Pd; //!< Bit-deletion probability \f$ P_d \f$
   double Pi; //!< Bit-insertion probability \f$ P_i \f$
   double Ps; //!< Probability of substitution error on non-inserted bits
   double Psi; //!< Probability of substitution error on inserted bits
   int T; //!< Block size in channel symbols over which we want to synchronize
   array1i_t Z; //!< Markov state sequence; Z(i) = drift after 'i' channel uses
   // @}
private:
   /*! \name Internal functions */
   void init();
   void precompute()
      {
      if (T > 0)
         computer.precompute(Pd, Pi, Ps, Psi, T, Zmin, Zmax);
      }
   void generate_state_sequence(const int tau);
   array1b_t generate_error_sequence();
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
   bpmr(const bool varyPd = true, const bool varyPi = true, const bool varyPs =
         false, const bool varyPsi = false);
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
   //! Set the substitution probability on non-inserted bits
   void set_ps(const double Ps)
      {
      assert(Ps >= 0 && Ps <= 0.5);
      this->Ps = Ps;
      precompute();
      }
   //! Set the substitution probability on inserted bits
   void set_psi(const double Psi)
      {
      assert(Psi >= 0 && Psi <= 0.5);
      this->Psi = Psi;
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
   //! Get the current substitution probability on non-inserted bits
   double get_ps() const
      {
      return Ps;
      }
   //! Get the current substitution probability on inserted bits
   double get_psi() const
      {
      return Psi;
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
      // state space is always fixed
      upper = Zmax;
      lower = Zmin;
      }
   //! Determine whether the channel model has a fixed state space
   bool is_statespace_fixed() const
      {
      return true;
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

   // Access to receiver metric computation object
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
