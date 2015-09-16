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

#ifndef __fba2_fss_h
#define __fba2_fss_h

#include "fba2-interface.h"
#include "matrix.h"
#include "multi_array.h"

#include <cmath>
#include <iostream>
#include <fstream>

namespace libcomm {

/*!
 * \brief   Symbol-Level Forward-Backward Algorithm (for TVB codes) with
 *          fixed state space (for BPMR channel).
 * \author  Johann Briffa
 *
 * Implements the forward-backward algorithm for a HMM, as required for the
 * MAP decoding algorithm for a generalized class of synchronization-correcting
 * codes described in
 * Johann A. Briffa, Victor Buttigieg, and Stephan Wesemeyer, "Time-varying
 * block codes for synchronisation errors: maximum a posteriori decoder and
 * practical issues. IET Journal of Engineering, 30 Jun 2014,
 * modified to work with channels with a fixed state space (as in the BPMR
 * channel of Iyangar & Wolf).
 *
 * \tparam receiver_t Type for receiver metric computer
 * \tparam sig Channel symbol type
 * \tparam real Floating-point type for internal computation
 * \tparam real2 Floating-point type for receiver metric computation
 * \tparam globalstore Flag indicating global pre-computation or caching of gamma values
 */

template <class receiver_t, class sig, class real, class real2, bool globalstore>
class fba2_fss : public fba2_interface<sig, real, real2> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<sig> array1s_t;
   typedef libbase::matrix<array1s_t> array2vs_t;
   typedef libbase::vector<double> array1d_t;
   typedef libbase::vector<real> array1r_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   typedef libbase::vector<array1r_t> array1vr_t;
   typedef boost::assignable_multi_array<real, 3> array3r_t;
   typedef boost::assignable_multi_array<real, 5> array5r_t;
   typedef boost::assignable_multi_array<real, 6> array6r_t;
   // @}
private:
   /*! \name Internally-used objects */
   mutable receiver_t receiver; //!< Inner code receiver metric computation
   array3r_t alpha; //!< Forward recursion metric, indices (i,m,delta)
   array3r_t beta; //!< Backward recursion metric, indices (i,m,delta)
   mutable struct {
      array6r_t global; // indices (i,m1,delta1,d,m2,delta2)
      array5r_t local; // indices (m1,delta1,d,m2,delta2)
   } gamma; //!< Receiver metric
   mutable array1i_t cw_length; //!< Codeword 'i' length
   mutable array1i_t cw_start; //!< Codeword 'i' start
   mutable int tau; //!< Frame length (all codewords in sequence)
   array1s_t r; //!< Copy of received sequence, for local computation of gamma
   array1vd_t app; //!< Copy of a-priori statistics, for local computation of gamma
   bool initialised; //!< Flag to indicate when memory is allocated
   // @}
   /*! \name User-defined parameters */
   int N; //!< The transmitted block size in symbols
   int q; //!< The number of symbols in the q-ary alphabet
   int Zmin; //!< The largest negative drift within a whole frame is \f$ m_\tau^{-} \f$
   int Zmax; //!< The largest positive drift within a whole frame is \f$ m_\tau^{+} \f$
   // @}
private:
   /*! \name Internal functions - computer */
   /*! \brief Get a reference to the corresponding gamma storage entry
    * This method is called from nested loops as follows:
    * - from work_alpha:
    *        i,m1,d=outer loops
    *        m2=inner loop
    * - from work_beta:
    *        i,m1,d=outer loops
    *        m2=inner loop
    * - from work_message_app:
    *        i,d,m1=outer loops
    *        m2=inner loop
    */
   real& gamma_storage_entry(int d, int i, int m1, int delta1, int m2, int delta2) const
      {
      if (globalstore)
         return gamma.global[i][m1][delta1][d][m2][delta2];
      else
         return gamma.local[m1][delta1][d][m2][delta2];
      }
   // common small tasks
   static real get_scale(const array3r_t& metric, int i, int m_min, int m_max);
   static void normalize(array3r_t& metric, int i, int m_min, int m_max);
   void normalize_alpha(int i)
      {
      normalize(alpha, i, Zmin, Zmax);
      }
   void normalize_beta(int i)
      {
      normalize(beta, i, Zmin, Zmax);
      }
   // decode functions - partial computations
   void work_gamma(const array1s_t& r, const array1vd_t& app, const int i) const
      {
      // allocate space for results
      static array1r_t ptable0, ptable1;
      ptable0.init(Zmax - Zmin + 1);
      ptable1.init(Zmax - Zmin + 1);
      // determine if this is the first or last codeword
      const bool first = (i == 0);
      const bool last = (i == N - 1);
      // for each start drift
      for (int m1 = Zmin; m1 <= Zmax; m1++)
         {
         // determine received segment to extract
         // n * i = offset to start of current codeword
         // -Zmin = offset to zero drift in 'r'
         // Zmax-m1 = maximum positive drift for a start drift of 'm1'
         // n = maximum positive drift over 'n' bits
         const int start = cw_start(i) + m1 - Zmin;
         const int length = std::min(cw_length(i) + std::min(Zmax - m1, cw_length(i)),
               r.size() - start);
         // for each initial deletion option
         for (int delta1 = 0; delta1 <= 1; delta1++)
            // for each symbol value
            for (int d = 0; d < q; d++)
               {
               // call batch receiver method
               receiver.R(d, i, r.extract(start, length), m1, delta1, first,
                     last, app, ptable0, ptable1);
               // store in corresponding place in storage
               for (int m2 = Zmin; m2 <= Zmax; m2++)
                  {
                  gamma_storage_entry(d, i, m1, delta1, m2, 0) = ptable0(
                        m2 - Zmin);
                  gamma_storage_entry(d, i, m1, delta1, m2, 1) = ptable1(
                        m2 - Zmin);
                  }
            }
         }
      }
   void work_alpha(const int i);
   void work_beta(const int i);
   void work_message_app(array1vr_t& ptable, const int i) const;
   void work_state_app(array1r_t& ptable, const int i) const;
   // @}
private:
   /*! \name Internal functions - main */
   // memory allocation
   void allocate();
   void free();
   // helper methods
   void print_gamma(std::ostream& sout) const;
   void print_metric(std::ostream& sout, const array3r_t& metric) const;
   void print_alpha(std::ostream& sout) const
      {
      sout << "alpha = " << std::endl;
      print_metric(sout, alpha);
      }
   void print_beta(std::ostream& sout) const
      {
      sout << "beta = " << std::endl;
      print_metric(sout, beta);
      }
   // decode functions - global path
   void work_gamma(const array1s_t& r, const array1vd_t& app);
   void work_alpha_and_beta(const array1d_t& sof_prior,
         const array1d_t& eof_prior);
   void work_results(array1vr_t& ptable, array1r_t& sof_post,
         array1r_t& eof_post) const;
   // decode functions - local path
   void work_alpha(const array1d_t& sof_prior);
   void work_beta_and_results(const array1d_t& eof_prior, array1vr_t& ptable,
         array1r_t& sof_post, array1r_t& eof_post);
   // @}
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   fba2_fss() :
         initialised(false)
      {
      }
   // @}

   /*! \name FBA2 Interface Implementation */
   /*! \brief Set up code size, decoding parameters, and channel receiver
    * Only needs to be done before the first frame.
    */
   void init(int N, int q, int mtau_min, int mtau_max, int mn_min, int mn_max,
         int m1_min, int m1_max, double th_inner, double th_outer,
         const typename libcomm::channel_insdel<sig, real2>::metric_computer& computer);
   /*! \brief Set up encoding table
    * Needs to be done before every frame.
    */
   void init(const array2vs_t& encoding_table) const
      {
      this->receiver.init(encoding_table);
      // Initialize arrays with start and length of each codeword
      cw_length.init(N);
      cw_start.init(N);
      int start = 0;
      for (int i = 0; i < N; i++)
         {
         const int n = encoding_table(i, 0).size();
         cw_start(i) = start;
         cw_length(i) = n;
         start += n;
         }
      tau = start;
      }

   // decode functions
   void decode(libcomm::instrumented& collector, const array1s_t& r,
         const array1d_t& sof_prior, const array1d_t& eof_prior,
         const array1vd_t& app, array1vr_t& ptable, array1r_t& sof_post,
         array1r_t& eof_post, const int offset);
   void get_drift_pdf(array1r_t& pdf, const int i) const
      {
      work_state_app(pdf, i);
      }
   void get_drift_pdf(array1vr_t& pdftable) const;

   //! Description
   std::string description() const
      {
      return "Symbol-level Forward-Backward Algorithm [Fixed State Space]";
      }
   // @}
};

} // end namespace

#endif
