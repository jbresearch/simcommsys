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
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
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
 * \note Unlike the BSID and QIDS channels, this model has no concept of stream
 * operation, as at the receiving end, a full sector will always be retrieved.
 */

class bpmr : public channel_insdel<bool> {
private:
   // Shorthand for class hierarchy
   typedef channel<bool> Base;
public:
   /*! \name Type definitions */
   typedef float real;
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
private:
   /*! \name Internal representation */
   double Pd; //!< Bit-deletion probability \f$ P_d \f$
   double Pi; //!< Bit-insertion probability \f$ P_i \f$
   array1i_t Z; //!< Markov state sequence; Z(i) = drift after 'i' channel uses
   // @}
private:
   /*! \name Internal functions */
   void init();
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
      }
   //! Set the bit-insertion probability
   void set_pi(const double Pi)
      {
      assert(Pi >= 0 && Pi <= 1);
      assert(Pi + Pd >= 0 && Pi + Pd <= 1);
      this->Pi = Pi;
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

   // Channel functions
   void transmit(const array1b_t& tx, array1b_t& rx);
   using Base::receive;
   void receive(const array1b_t& tx, const array1b_t& rx, array1vd_t& ptable) const
      {
      failwith("Method not defined.");
      }
   double receive(const array1b_t& tx, const array1b_t& rx) const
      {
      failwith("Method not defined.");
      return 0;
      }
   double receive(const bool& tx, const array1b_t& rx) const
      {
      failwith("Method not defined.");
      return 0;
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
