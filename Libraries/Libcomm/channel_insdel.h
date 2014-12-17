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

#ifndef __channel_insdel_h
#define __channel_insdel_h

#include "channel.h"
#include "clonable.h"
#include "cuda-all.h"

namespace libcomm {

/*!
 * \brief   Insertion-Deletion Channel Interface.
 * \author  Johann Briffa
 *
 * Defines the additional interface methods for insertion-deletion channels.
 *
 * \tparam S Channel symbol type
 * \tparam real Floating-point type for metric computer interface
 */

template <class S, class real>
class channel_insdel : public channel<S> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<real> array1r_t;
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<S> array1s_t;
   // @}
public:
   /*! \name Metric computation */
   class metric_computer : public libbase::clonable {
   public:
#ifdef USE_CUDA
      /*! \name Device methods */
#ifdef __CUDACC__
      //! Receiver interface
      __device__
      virtual real receive(const cuda::vector_reference<S>& tx, const cuda::vector_reference<S>& rx) const = 0;
      //! Batch receiver interface
      __device__
      virtual void receive(const cuda::vector_reference<S>& tx, const cuda::vector_reference<S>& rx,
            cuda::vector_reference<real>& ptable) const = 0;
#endif
      // @}
#endif
      /*! \name Host methods */
      //! Determine the amount of shared memory required per thread
      virtual size_t receiver_sharedmem() const = 0;
      //! Batch receiver interface - indefinite state space
      virtual void receive(const array1s_t& tx, const array1s_t& rx,
            array1r_t& ptable) const = 0;
      //! Batch receiver interface - fixed state space
      virtual void receive(const array1s_t& tx, const array1s_t& rx,
            const int S0, const int delta0, const bool first, const bool last,
            array1r_t& ptable0, array1r_t& ptable1) const = 0;
      // @}
   };
   // @}
public:
   /*! \name Insertion-deletion channel functions */
   /*!
     * \brief Get the actual channel drift at time 't' of last transmitted frame.
     */
   virtual int get_drift(int t) const = 0;
   /*!
     * \brief Get the actual channel drift at a set of times 't' of last transmitted frame.
     */
   virtual array1i_t get_drift(const array1i_t& t) const
       {
       // allocate space for results
       array1i_t result(t.size());
       // consider each time index in the order given
       for (int i = 0; i < t.size(); i++)
          result(i) = get_drift(t(i));
       return result;
       }
   //! Set the probability of channel event outside chosen limits
   virtual void set_pr(const double Pr) = 0;
   //! Set the block size
   virtual void set_blocksize(int T) = 0;
   //! Determine whether the channel model has a fixed state space
   virtual bool is_statespace_fixed() const = 0;
   /*!
    * \brief Determine upper and lower drift limits at the end of a frame of
    * 'T' symbols, given the supplied drift pdf at start of transmission.
    */
   virtual void compute_limits(int tau, double Pr, int& lower, int& upper,
         const libbase::vector<double>& sof_pdf = libbase::vector<double>(),
         const int offset = 0) const = 0;
   // @}

   /*! \name Access to receiver metric computation object */
   virtual const metric_computer& get_computer() const = 0;
   // @}
};

} // end namespace

#endif
