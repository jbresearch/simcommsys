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

#ifndef __qids_utils_h
#define __qids_utils_h

#include "config.h"
#include "matrix.h"
#include <cmath>
#include <limits>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show results of computation of xmax and I
// 3 - Show intermediate computation details for (2)
// NOTE: since this is a header, it may be included in other classes as well;
//       to avoid problems, the debug level is reset at the end of this file.
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*!
 * \brief   Helper class for q-ary insertion/deletion/substitution channel.
 * \author  Johann Briffa
 *
 * This is a helper class for the q-ary insertion, deletion and substitution
 * error channel (qids). It implements (as static methods) a number of
 * functions for the computation of the channel drift distribution and the
 * dependent state space limits.
 */

class qids_utils {
private:
   /*! \name Internal class definitions */
   //! Functor for drift probability computation with prior
   class compute_drift_prob_functor {
   public:
      typedef double (*pdf_func_t)(int, int, double, double);
   private:
      const pdf_func_t func;
      const libbase::vector<double>& sof_pdf;
      const int offset;
   public:
      compute_drift_prob_functor(const pdf_func_t& func,
            const libbase::vector<double>& sof_pdf, const int offset) :
            func(func), sof_pdf(sof_pdf), offset(offset)
         {
         }
      double operator()(int m, int T, double Pi, double Pd) const
         {
         return compute_drift_prob_with(func, m, T, Pi, Pd, sof_pdf, offset);
         }
   };
   /*! \brief A class representing a pair of integers.
    * Used to represent the key value in a std::map used to cache results of
    * compute_drift_prob_exact.
    */
   class pair {
   public:
      int a;
      int b;
   public:
      pair(const int& a, const int& b) :
            a(a), b(b)
         {
         }
   };
   /*! \brief Functor for comparing pairs.
    * \note This makes use of short-circuit evaluation
    */
   struct compare_pairs {
      bool operator()(const pair& lhs, const pair& rhs) const
         {
         return lhs.a < rhs.a || (lhs.a == rhs.a && lhs.b < rhs.b);
         }
   };
   // @}
   /*! \name Internal function definitions */
   static double compute_drift_prob_exact_0(int m, int T, double Pi, double Pd);
   static double compute_drift_prob_exact_1(int m, int T, double Pi, double Pd);
   static double compute_drift_prob_exact_2(int m, int T, double Pi, double Pd);
   static double compute_drift_prob_exact_3(int m, int T, double Pi, double Pd);
   // @}
public:
   /*! \name Channel drift computation */
   static double compute_drift_prob_davey(int m, int T, double Pi, double Pd);
   static double compute_drift_prob_exact(int m, int T, double Pi, double Pd);
   /*!
    * \brief Computes the probability of drift 'm' after transmitting 'T' symbols
    * using the supplied algorithm and drift pdf at start of transmission.
    *
    * The final drift pdf is obtained by convolving the expected pdf for a
    * known start of frame position with the actual start of frame distribution.
    */
   template <typename F>
   static double compute_drift_prob_with(const F& compute_pdf, int m, int T,
         double Pi, double Pd, const libbase::vector<double>& sof_pdf,
         const int offset)
      {
      // if sof_pdf is empty, delegate automatically
      if (sof_pdf.size() == 0)
         return compute_pdf(m, T, Pi, Pd);
      // compute the probability at requested drift
      double this_p = 0;
      const int imin = -offset;
      const int imax = sof_pdf.size() - offset;
      for (int i = imin; i < imax; i++)
         {
         const double p = compute_pdf(m - i, T, Pi, Pd);
         this_p += sof_pdf(i + offset) * p;
         }
      // normalize
      this_p /= sof_pdf.sum();
      // confirm that this value is finite and valid
      assert(this_p >= 0 && this_p < std::numeric_limits<double>::infinity());
      return this_p;
      }
   // @}

   /*! \name State space limits computation - approximations */
   // limit on successive insertions
   static int compute_I(int T, double Pi, double Pr);
   // limit on drift
   static int compute_xmax_davey(int T, double Pi, double Pd, double Pr);
   // @}

   /*! \name State space limits computation - low level functions */
   /*!
    * \brief Determine the probability of the drift at the end of a frame
    * of T channel symbols being outside the upper and lower limits, using the
    * supplied algorithm for computing the drift pdf.
    */
   template <typename F>
   static double compute_outofbounds_with(const F& compute_pdf, int T,
         double Pi, double Pd, int upper, int lower)
      {
      // sanity checks
      assert(T > 0);
      validate(Pd, Pi);
      assert(upper >= 0);
      assert(lower <= 0);
      // determine area that needs to be covered
      double coverage = 1.0;
      // subtract area that is covered, starting from center
      coverage -= compute_pdf(0, T, Pi, Pd);
      const int m1 = std::min(upper, -lower) + 1;
      for (int m = 1; m < m1; m++)
         {
         coverage -= compute_pdf(m, T, Pi, Pd);
         coverage -= compute_pdf(-m, T, Pi, Pd);
         }
      for (int m = m1; m <= upper; m++)
         coverage -= compute_pdf(m, T, Pi, Pd);
      for (int m = -m1; m >= lower; m--)
         coverage -= compute_pdf(m, T, Pi, Pd);
      // return result
      return coverage;
      }
   /*!
    * \brief Determine separate upper/lower limits for drift at the end of a
    * frame of \f$ T \f$ channel symbols, using the supplied algorithm for
    * computing the drift pdf.
    *
    * The drift range is chosen such that the probability of having a drift
    * \f$ m \f$ after transmitting \f$ T \f$ symbols, \f$ \phi_T(m) \f$,
    * is less than an arbitrary value \f$ \frac{P_r}{2} \f$ for any
    * \f$ m \f$ outside the given limit.
    */
   template <typename F>
   static void compute_limits_with(const F& compute_pdf, int T, double Pi,
         double Pd, double Pr, int& mT_min, int& mT_max)
      {
      // sanity checks
      assert(T > 0);
      validate(Pd, Pi);
      // keep track of coverage
      double coverage = 1.0;
      coverage -= compute_pdf(0, T, Pi, Pd);
      // determine lower limit first
      double p_lower;
      for (int m = -1;; m--)
         {
         p_lower = compute_pdf(m, T, Pi, Pd);
         if (p_lower < Pr / 2)
            {
            mT_min = m + 1;
            break;
            }
         coverage -= p_lower;
         }
      // next determine upper limit
      double p_upper;
      for (int m = 1;; m++)
         {
         p_upper = compute_pdf(m, T, Pi, Pd);
         if (p_upper < Pr / 2)
            {
            mT_max = m - 1;
            break;
            }
         coverage -= p_upper;
         }
      // now fine-tune the selection
      while (coverage >= Pr)
         {
         // extend in the direction of the largest gain
         if (p_upper > p_lower)
            {
            mT_max++;
            coverage -= p_upper;
            // note: p_upper always corresponds to next higher state
            p_upper = compute_pdf(mT_max + 1, T, Pi, Pd);
            }
         else
            {
            mT_min--;
            coverage -= p_lower;
            // note: p_lower always corresponds to next lower state
            p_lower = compute_pdf(mT_min - 1, T, Pi, Pd);
            }
         }
      }
   /*!
    * \brief Determine maximum absolute drift at the end of a frame of \f$ T \f$
    * symbols, using the supplied algorithm for computing the drift pdf.
    *
    * The drift range is chosen such that the probability of having the drift
    * after transmitting \f$ T \f$ symbols being greater than \f$ \pm x_{max} \f$
    * is less than an arbitrary value \f$ P_r \f$.
    */
   template <typename F>
   static int compute_xmax_with(const F& compute_pdf, int T, double Pi,
         double Pd, double Pr)
      {
      // sanity checks
      assert(T > 0);
      validate(Pd, Pi);
      // determine area that needs to be covered
      double acc = 1.0;
      // determine xmax to use
      int xmax = 0;
      acc -= compute_pdf(xmax, T, Pi, Pd);
#if DEBUG>=3
      std::cerr << "DEBUG (qids): xmax = " << xmax << ", acc = " << acc << "." << std::endl;
#endif
      while (acc >= Pr)
         {
         xmax++;
         acc -= compute_pdf(xmax, T, Pi, Pd);
         acc -= compute_pdf(-xmax, T, Pi, Pd);
#if DEBUG>=3
         std::cerr << "DEBUG (qids): xmax = " << xmax << ", acc = " << acc << "." << std::endl;
#endif
         }
      // tell the user what we did and return
#if DEBUG>=2
      std::cerr << "DEBUG (qids): [computed] for T = " << T << ", xmax = " << xmax << "." << std::endl;
      std::cerr << "DEBUG (qids): [davey] for T = " << T << ", xmax = " << compute_xmax_davey(T, Pi, Pd, Pr) << "." << std::endl;
#endif
      return xmax;
      }
   // @}

   /*! \name State space limits computation - high level functions */
   /*!
    * \brief Determine maximum drift at the end of a frame of 'T' symbols, given
    * the supplied drift pdf at start of transmission.
    */
   static int compute_xmax(int T, double Pi, double Pd, double Pr,
         const libbase::vector<double>& sof_pdf = libbase::vector<double>(),
         const int offset = 0)
      {
      compute_drift_prob_functor f(compute_drift_prob_exact, sof_pdf, offset);
      const int xmax = compute_xmax_with(f, T, Pi, Pd, Pr);
#if DEBUG>=3
      std::cerr << "DEBUG (qids): [exact] for T = " << T << ", xmax = " << xmax << "." << std::endl;
#endif
      return xmax;
      }
   /*!
    * \brief Determine upper and lower drift limits at the end of a frame of
    * 'T' symbols, given the supplied drift pdf at start of transmission.
    */
   static void compute_limits(int T, double Pi, double Pd, double Pr,
         int& mT_min, int& mT_max,
         const libbase::vector<double>& sof_pdf = libbase::vector<double>(),
         const int offset = 0)
      {
      compute_drift_prob_functor f(compute_drift_prob_exact, sof_pdf, offset);
      compute_limits_with(f, T, Pi, Pd, Pr, mT_min, mT_max);
#if DEBUG>=3
      std::cerr << "DEBUG (qids): [exact] for T = " << T << ", mT_min = " << mT_min << ", mT_max = " << mT_max << "." << std::endl;
#endif
      }
   // @}

   /*! \name General utility functions */
   //! Check validity of Pi and Pd
   static void validate(double Pd, double Pi)
      {
      assert(Pi >= 0 && Pi < 1.0);
      assert(Pd >= 0 && Pd < 1.0);
      assert(Pi + Pd >= 0 && Pi + Pd < 1.0);
      }
   /*! Determines the probability of error over multiple independent events
    * from the individual error probability.
    *
    * \param Pe The probability of error of an individual event
    * \param events The number of events liable to independent error
    * \return The equivalent probability over multiple independent events
    */
   static double multiply_error_probability(double Pe, int events)
      {
      return 1 - pow(1-Pe, events);
      }
   /*! Determines the probability of an individual error event from the
    * probability over multiple independent events.
    *
    * \param Pe The probability of error over the set of multiple events
    * \param events The number of events liable to independent error
    * \return The equivalent probability of each independent error
    */
   static double divide_error_probability(double Pe, int events)
      {
      return 1 - pow(1-Pe, 1/double(events));
      }
   // @}
};

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG
#endif

} // end namespace

#endif
