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

#include "qids-utils.h"
#include "itfunc.h"
#include <exception>
#include <cmath>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show results of computation of xmax and I
// 3 - Show intermediate computation details for (2)
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

// FBA decoder parameter computation

/*!
 * \brief Computes the probability of drift 'm' after transmitting 'T' symbols
 * using Davey's Gaussian approximation.
 *
 * The calculation is based on the assumption that the end-of-frame drift has
 * a Gaussian distribution with zero mean and standard deviation given by
 * \f$ \sigma = \sqrt{\frac{2 T p}{1-p}} \f$
 * where \f$ p = P_i = P_d \f$.
 *
 * To preserve the property that the sum over x should be 1, rather than
 * returning a discrete sample of the Gaussian pdf at \f$ m \f$, we return the
 * integral over \f$ m \pm 0.5 \f$.
 *
 * Since the probability is symmetric about \f$ m=0 \f$, we always use positive
 * \f$ m \f$; this ensures the subtraction does not fail due to resolution of
 * double.
 */
double qids_utils::compute_drift_prob_davey(int m, int T, double Pi, double Pd)
   {
   // sanity checks
   assert(T > 0);
   validate(Pd, Pi);
   assertalways(Pi == Pd);
   // the distribution is approximately Gaussian with:
   const double p = Pi;
   const double sigma = sqrt(2 * T * p / (1 - p));
   // main computation
   const double xa = abs(m);
   const double x1 = (xa - 0.5) / sigma;
   const double x2 = (xa + 0.5) / sigma;
   const double this_p = libbase::Q(x1) - libbase::Q(x2);
#if DEBUG>=3
   std::cerr << "DEBUG (qids-utils): [pdf-davey] m = " << m << ", sigma = " << sigma << ", this_p = " << this_p << "." << std::endl;
#endif
   return this_p;
   }

/*! \copydoc qids_utils::compute_drift_prob_exact()
 *
 * Non-degenerate case: Pi > 0, Pd > 0
 *
 * The required probability Pr{ S_T = m } is calculated as:
 *
 * sum for j=j0..T of
 *
 *    p_j = Pt^T Pi^m Pf^j C(T, j) C(T+m+j-1, m+j)
 *
 * where j0 = max(-m,0),
 *       Pf = Pi Pd / Pt,
 *       Pt = 1 - Pi - Pd,
 * and the binomial coefficient is expressed as
 *    C(n,k) = n! / [ k! (n-k)! ] for k ≤ n
 *           = n(n-1)...(n-k-1) / k(k-1)...1
 *           = 0 for k > n
 *
 * Note that:
 * a) the summation is empty if j0 > T, resulting in zero probability
 * b) the first binomial coefficient is always non-zero
 * c) the second b.c. is non-zero if T-1 >= 0, or equivalently T > 0
 *
 * By expanding the binomial coefficients using factorials, note also that:
 *
 *    p_j = p_{j-1} . Pf . (T+m+j-1) . (T-j+1)
 *                           (m+j)        j
 *
 * This allows successive factors to be determined easily from previous ones.
 * The initial factor required is the one at j0, determined as:
 *
 *    p_j0 = Pt^T . Pi^m . Pf^j0 . C(T, j0) . C(T+m+j0-1, m+j0)
 *
 * Now the binomial coefficients can be computed using the multiplicative
 * formula:
 *    C(n,k) = product for i=1..k of (n-k-i)/i
 *
 * Therefore the binomial coefficients in p_j0 are computed as:
 *
 * C(T, j0) = product for i=1..j0 of (T-j0-i)/i
 * C(T+m+j0-1, m+j0) = product for i=1..m+j0 of (T-1-i)/i
 */
double qids_utils::compute_drift_prob_exact_0(int m, int T, double Pi, double Pd)
   {
   assert(Pi > 0 && Pd > 0);
   typedef long double myreal;
   // shortcut for out-of-range values
   if (m < -T) // too many deletions required
      return 0;
   // set constants
   const double Pt = 1 - Pi - Pd;
   const double Pf = Pi * Pd / Pt;
   const int j0 = std::max(-m, 0);
   // compute common factor (in log domain) for j=j0
   myreal pj = 0;
   // include first two terms in p_j0
   pj += log(myreal(Pt)) * T;
   pj += log(myreal(Pi)) * m;
   // include third term in p_j0
   if (j0 > 0)
      pj += log(Pf) * j0;
   // include first binomial coefficient term in p_j0
   for (int i = 1; i <= j0; i++)
      pj += log(myreal(T - j0 + i)) - log(myreal(i));
   // include second binomial coefficient term in p_j0
   for (int i = 1; i <= m + j0; i++)
      pj += log(myreal(T - 1 + i)) - log(myreal(i));
#if DEBUG>=3
   std::cerr << "DEBUG (qids-utils): [pdf-exact] m = " << m << ", p_j0 = " << pj << std::endl;
#endif
   // main computation
   myreal this_p = exp(pj);
   for (int j = j0 + 1; j <= T; j++)
      {
      // update factor
      pj += log(Pf);
      pj += log(myreal(T + m + j - 1)) - log(myreal(m + j));
      pj += log(myreal(T - j + 1)) - log(myreal(j));
      // update main result
      this_p += exp(pj);
#if DEBUG>=3
      std::cerr << "DEBUG (qids-utils): [pdf-exact] j = " << j << ", p_j = " << pj << ", this_p = " << this_p << std::endl;
#endif
      }
   return this_p;
   }

/*! \copydoc qids_utils::compute_drift_prob_exact()
 *
 * Degenerate case 1: Pi > 0, Pd = 0
 *    p_j is non-zero only for j=0, so that:
 *    Pr{ S_T = m } = Pt^T Pi^m C(T+m-1, m) for m >= 0,
 *                  = 0 otherwise
 */
double qids_utils::compute_drift_prob_exact_1(int m, int T, double Pi, double Pd)
   {
   assert(Pi > 0 && Pd == 0);
   typedef long double myreal;
   // shortcut for out-of-range values
   if (m < 0) // deletions required
      return 0;
   // set constants
   const double Pt = 1 - Pi - Pd;
   // compute result in log domain
   myreal result = 0;
   // include first two terms in result
   result += log(myreal(Pt)) * T;
   result += log(myreal(Pi)) * m;
   // include binomial coefficient term in result
   for (int i = 1; i <= m; i++)
      result += log(myreal(T - 1 + i)) - log(myreal(i));
   // convert factor back from log domain
   return exp(result);
   }

/*! \copydoc qids_utils::compute_drift_prob_exact()
 *
 * Degenerate case 2: Pd > 0, Pi = 0
 *    p_j is non-zero only for j=-m, so that:
 *    Pr{ S_T = m } = Pt^{T+m} Pd^-m C(T, -m) for m <= 0,
 *                  = 0 otherwise
 */
double qids_utils::compute_drift_prob_exact_2(int m, int T, double Pi, double Pd)
   {
   assert(Pi == 0 && Pd > 0);
   typedef long double myreal;
   // shortcut for out-of-range values
   if (m > 0) // insertions required
      return 0;
   else if (m < -T) // too many deletions required
      return 0;
   // set constants
   const double Pt = 1 - Pi - Pd;
   // compute result in log domain
   myreal result = 0;
   // include first two terms in result
   result += log(myreal(Pt)) * (T + m);
   result += log(myreal(Pd)) * (-m);
   // include binomial coefficient term in result
   for (int i = 1; i <= -m; i++)
      result += log(myreal(T + m + i)) - log(myreal(i));
   // convert factor back from log domain
   return exp(result);
   }

/*! \copydoc qids_utils::compute_drift_prob_exact()
 *
 * Degenerate case 3: Pi = Pd = 0
 *    p_j is non-zero only for j=0 and m=0, so that:
 *    Pr{ S_T = m } = 1 for m = 0,
 *                  = 0 otherwise
 */
double qids_utils::compute_drift_prob_exact_3(int m, int T, double Pi, double Pd)
   {
   assert(Pi == 0 && Pd == 0);
   return (m == 0) ? 1 : 0;
   }

/*!
 * \brief Computes the probability of drift 'm' after transmitting 'T' symbols
 * using the exact metric from our submission to Transactions on Communications.
 */
double qids_utils::compute_drift_prob_exact(int m, int T, double Pi, double Pd)
   {
   typedef long double myreal;
   // sanity checks
   assert(T > 0);
   validate(Pd, Pi);
#if DEBUG>=3
   std::cerr << "DEBUG (qids-utils): compute_drift_prob_exact(" << m << "," << T << "," << Pi << "," << Pd << ")" << std::endl;
#endif
   // caching of results
   static std::map<pair, double, compare_pairs> cache;
   static double last_Pi = -1; // initialize to an invalid value
   static double last_Pd = -1; // initialize to an invalid value
   if (last_Pi != Pi || last_Pd != Pd)
      {
      last_Pi = Pi;
      last_Pd = Pd;
      cache.clear();
      }
   // see if we have this value already in cache
   const pair key(m,T);
   std::map<pair, double, compare_pairs>::const_iterator it = cache.find(key);
   // return it if we do
   if (it != cache.end())
      return it->second;
   // space for result
   myreal this_p;
   // handle non-degenerate case: Pi > 0, Pd > 0
   if (Pi > 0 && Pd > 0)
      this_p = compute_drift_prob_exact_0(m, T, Pi, Pd);
   // handle degenerate case 1: Pi > 0, Pd = 0
   else if (Pi > 0)
      this_p = compute_drift_prob_exact_1(m, T, Pi, Pd);
   // handle degenerate case 2: Pd > 0, Pi = 0
   else if (Pd > 0)
      this_p = compute_drift_prob_exact_2(m, T, Pi, Pd);
   // handle degenerate case 3: Pi = Pd = 0
   else
      this_p = compute_drift_prob_exact_3(m, T, Pi, Pd);
#if DEBUG>=3
   std::cerr << "DEBUG (qids-utils): [pdf-exact] this_p = " << this_p << std::endl;
#endif
   // validate and return result
   if (!std::isfinite(this_p))
      throw std::overflow_error("value not finite");
   else if (this_p < 0)
      throw std::overflow_error("negative value");
   // store this value in cache
   cache[key] = this_p;
   return this_p;
   }

/*!
 * \brief Determine limit for insertions between two time-steps.
 *
 * \f[ I = \left\lceil \frac{ \log{P_r} - \log T }{ \log P_i } \right\rceil - 1 \f]
 * where \f$ P_r \f$ is an arbitrary probability of having a block of size
 * \f$ T \f$ with at least one event of more than \f$ I \f$ insertions
 * between successive time-steps.
 *
 * \note For \f$ P_i > 0 \f$, the smallest allowed value is \f$ I = 1 \f$.
 */
int qids_utils::compute_I(int T, double Pi, double Pr)
   {
   // sanity checks
   assert(T > 0);
   assert(Pi >= 0 && Pi < 1.0);
   // shortcut for no-insertion case
   if (Pi == 0)
      return 0;
   // main computation
   int I = int(ceil((log(Pr) - log(double(T))) / log(Pi))) - 1;
   I = std::max(I, 1);
#if DEBUG>=2
   std::cerr << "DEBUG (qids-utils): for T = " << T << ", I = " << I << std::endl;
#endif
   return I;
   }

/*!
 * \brief Determine maximum absolute drift at the end of a frame of 'T' symbols
 * using Davey's Gaussian approximation.
 *
 * \f[ x_{max} = Q^{-1}(\frac{P_r}{2}) \sqrt{\frac{T p}{1-p}} \f]
 * where \f$ p = P_i = P_d \f$ and \f$ P_r \f$ is an arbitrary probability of
 * having a block of size \f$ T \f$ where the drift at the end is greater
 * than \f$ \pm x_{max} \f$.
 *
 * The calculation is based on the assumption that the end-of-frame drift has
 * a Gaussian distribution with zero mean and standard deviation given by
 * \f$ \sigma = \sqrt{\frac{2 T p}{1-p}} \f$.
 */
int qids_utils::compute_xmax_davey(int T, double Pi, double Pd, double Pr)
   {
   // sanity checks
   assert(T > 0);
   validate(Pd, Pi);
   assertalways(Pi == Pd);
   // determine required multiplier
   const double factor = libbase::Qinv(Pr / 2.0);
#if DEBUG>=3
   std::cerr << "DEBUG (qids-utils): [davey] Q(" << factor << ") = " << libbase::Q(
         factor) << std::endl;
#endif
   // main computation
   const double p = Pi;
   const int xmax = int(ceil(factor * sqrt(2 * T * p / (1 - p))));
   // tell the user what we did and return
   return xmax;
   }

} // end namespace
