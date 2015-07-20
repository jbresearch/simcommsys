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

#ifndef __dids_h
#define __dids_h

#include "config.h"
#include "bitfield.h"
#include "channel_insdel.h"
#include "serializer.h"
#include "matrix.h"
#include <cmath>

#include "bpmr.h"

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
 * \brief   Data-dependent Insertion, Deletion and Substitution (DIDS) channel.
 * \author  Johann Briffa
 *
 * This class implements the recording channel for bit-patterned media
 * described in:
 * Tong Wu and Marc A. Armand, "The Davey-MacKay Coding Scheme for Channels
 * With Dependent Insertion, Deletion, and Substitution Errors,"
 * IEEE Transactions on Magnetics, vol.49, no.1, pp.489-495, Jan. 2013
 * URL: http://dx.doi.org/10.1109/TMAG.2012.2208120
 *
 * \tparam real Floating-point type for internal computation
 *
 * \note Like the BPMR channel, this model has no concept of stream operation,
 * as at the receiving end, a full sector will always be retrieved.
 *
 * \todo implement receiver
 */

template <class real>
class dids : public channel_insdel<bool, real> {
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
   bool varyPd; //!< Flag to indicate that \f$ P_D \f$ should change with parameter
   bool varyPi; //!< Flag to indicate that \f$ P_I \f$ should change with parameter
   bool varyPr; //!< Flag to indicate that \f$ P_R \f$ should change with parameter
   bool varyPb; //!< Flag to indicate that \f$ P_B \f$ should change with parameter
   int Zmax; //!< Maximum value of Markov state (equal to K-1 in Iyengar paper)
   int Zmin; //!< Minimum value of Markov state (0 for Iyengar model, otherwise negative)
   int L; //!< Length of error burst from location of synchronization error
   double fixedPd; //!< Value to use when \f$ P_D \f$ does not change with parameter
   double fixedPi; //!< Value to use when \f$ P_I \f$ does not change with parameter
   double fixedPr; //!< Value to use when \f$ P_R \f$ does not change with parameter
   double fixedPb; //!< Value to use when \f$ P_B \f$ does not change with parameter
   // @}
private:
   /*! \name Internal representation */
   typename bpmr<real>::metric_computer computer;
   double Pd; //!< Bit-deletion probability \f$ P_D \f$
   double Pi; //!< Bit-insertion probability \f$ P_I \f$
   double Pr; //!< Probability of random substitution error
   double Pb; //!< Probability of burst substitution error
   int T; //!< Block size in channel symbols over which we want to synchronize
   array1i_t Z; //!< Markov state sequence; Z(i) = drift after 'i' channel uses
   // @}
private:
   /*! \name Internal functions */
   void init();
   void precompute()
      {
      if (T > 0)
         {
         // estimate average substitution error rate
         const double Pid = 2 * L * (Pi + Pd);
         const double Ps = Pr * (1 - Pid) + Pb * Pid;
         computer.precompute(Pd, Pi, Ps, Pb, T, Zmin, Zmax);
         }
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
   dids(const bool varyPd = true, const bool varyPi = true, const bool varyPr =
         false, const bool varyPb = false);
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
   //! Set the random substitution probability
   void set_ps(const double Pr)
      {
      assert(Pr >= 0 && Pr <= 0.5);
      this->Pr = Pr;
      precompute();
      }
   //! Set the burst substitution probability
   void set_pb(const double Pb)
      {
      assert(Pb >= 0 && Pb <= 0.5);
      this->Pb = Pb;
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
   //! Get the current random substitution probability
   double get_ps() const
      {
      return Pr;
      }
   //! Get the current burst substitution probability
   double get_pb() const
      {
      return Pb;
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
   //! \note Used by dids::receive(tx, rx, ptable)
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
DECLARE_SERIALIZER(dids)
};

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG
#endif

} // end namespace

#endif
