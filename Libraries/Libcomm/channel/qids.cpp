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

#include "qids.h"
#include "itfunc.h"
#include <sstream>
#include <exception>
#include <cmath>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show transmit and insertion state vectors during transmission process
// 3 - Show results of computation of xmax and I
//     and details of pdf resizing
// 4 - Show intermediate computation details for (3)
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

// FBA decoder parameter computation

/*!
 * \brief The probability of drift x after transmitting tau symbols
 *
 * Computes the required probability using Davey's Gaussian approximation.
 *
 * The calculation is based on the assumption that the end-of-frame drift has
 * a Gaussian distribution with zero mean and standard deviation given by
 * \f$ \sigma = \sqrt{\frac{2 \tau p}{1-p}} \f$
 * where \f$ p = P_i = P_d \f$.
 *
 * To preserve the property that the sum over x should be 1, rather than
 * returning a discrete sample of the Gaussian pdf at x, we return the
 * integral over x ± 0.5.
 *
 * Since the probability is symmetric about x=0, we always use positive x;
 * this ensures the subtraction does not fail due to resolution of double.
 */
template <class G, class real>
double qids<G, real>::metric_computer::compute_drift_prob_davey(int x, int tau,
      double Pi, double Pd)
   {
   // sanity checks
   assert(tau > 0);
   validate(Pd, Pi);
   // set constants
   assertalways(Pi == Pd);
   // assumed by this algorithm
   const double p = Pi;
   // the distribution is approximately Gaussian with:
   const double sigma = sqrt(2 * tau * p / (1 - p));
   // main computation
   const double xa = abs(x);
   const double x1 = (xa - 0.5) / sigma;
   const double x2 = (xa + 0.5) / sigma;
   const double this_p = libbase::Q(x1) - libbase::Q(x2);
#if DEBUG>=4
   std::cerr << "DEBUG (qids): [pdf-davey] x = " << x << ", sigma = " << sigma
   << ", this_p = " << this_p << "." << std::endl;
#endif
   return this_p;
   }

/*!
 * \brief The probability of drift 'm' after transmitting 'T' symbols
 *
 * Computes the required probability using the exact metric from our
 * Trans. Inf. Theory submission.
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
 *
 * Degenerate cases:
 *
 * 1) Pi > 0, Pd = 0
 *    p_j is non-zero only for j=0, so that:
 *    Pr{ S_T = m } = Pt^T Pi^m C(T+m-1, m) for m >= 0,
 *                  = 0 otherwise
 *
 * 2) Pd > 0, Pi = 0
 *    p_j is non-zero only for j=-m, so that:
 *    Pr{ S_T = m } = Pt^{T+m} Pd^-m C(T, -m) for m <= 0,
 *                  = 0 otherwise
 *
 * 3) Pi = Pd = 0
 *    p_j is non-zero only for j=0 and m=0, so that:
 *    Pr{ S_T = m } = 1 for m = 0,
 *                  = 0 otherwise
 */
template <class G, class real>
double qids<G, real>::metric_computer::compute_drift_prob_exact(int m, int T,
      double Pi, double Pd)
   {
   typedef long double myreal;
   // sanity checks
   assert(T > 0);
   validate(Pd, Pi);
   // shortcut for out-of-range values (too many deletions required)
   if (m < -T)
      return 0;
   // set constants
   const double Pt = 1 - Pi - Pd;
   // handle degenerate case 3: Pi = Pd = 0
   if (Pi == 0 && Pd == 0)
      {
      return (m == 0) ? 1 : 0;
      }
   // handle degenerate case 2: Pd > 0, Pi = 0
   else if (Pi == 0)
      {
      // shortcut for out-of-range values (insertions required)
      if (m > 0)
         return 0;
      // compute result in log domain
      myreal result = 0;
      // include first two terms in result
      result += log(myreal(Pt)) * (T+m);
      result += log(myreal(Pd)) * (-m);
      // include binomial coefficient term in result
      for (int i = 1; i <= -m; i++)
         {
         result += log(myreal(T + m + i)) - log(myreal(i));
         }
      // convert factor back from log domain
      return exp(result);
      }
   // handle degenerate case 1: Pi > 0, Pd = 0
   else if (Pd == 0)
      {
      // shortcut for out-of-range values (deletions required)
      if (m < 0)
         return 0;
      // compute result in log domain
      myreal result = 0;
      // include first two terms in result
      result += log(myreal(Pt)) * T;
      result += log(myreal(Pi)) * m;
      // include binomial coefficient term in result
      for (int i = 1; i <= m; i++)
         {
         result += log(myreal(T - 1 + i)) - log(myreal(i));
         }
      // convert factor back from log domain
      return exp(result);
      }
   // set constants
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
      {
      pj += log(myreal(T - j0 + i)) - log(myreal(i));
      }
   // include second binomial coefficient term in p_j0
   for (int i = 1; i <= m + j0; i++)
      {
      pj += log(myreal(T - 1 + i)) - log(myreal(i));
      }
#if DEBUG>=4
   std::cerr << "DEBUG (qids): [pdf-exact] m = " << m << ", p_j0 = " << pj
   << std::endl;
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
#if DEBUG>=4
      std::cerr << "DEBUG (qids): [pdf-exact] j = " << j << ", p_j = " << pj
      << ", this_p = " << this_p << std::endl;
#endif
      }
#if DEBUG>=4
   std::cerr << "DEBUG (qids): [pdf-exact] this_p = " << this_p << std::endl;
#endif
   // validate and return result
   if (!std::isfinite(this_p))
      throw std::overflow_error("value not finite");
   else if (this_p < 0)
      throw std::overflow_error("negative value");
   return this_p;
   }

/*!
 * \brief Determine limit for insertions between two time-steps
 * 
 * \f[ I = \left\lceil \frac{ \log{P_r} - \log \tau }{ \log P_i } \right\rceil - 1 \f]
 * where \f$ P_r \f$ is an arbitrary probability of having a block of size
 * \f$ \tau \f$ with at least one event of more than \f$ I \f$ insertions
 * between successive time-steps.
 * In this class, this value is fixed.
 * 
 * \note The smallest allowed value is \f$ I = 1 \f$; the largest value depends
 * on a user parameter.
 */
template <class G, class real>
int qids<G, real>::metric_computer::compute_I(int tau, double Pi, double Pr, int Icap)
   {
   // sanity checks
   assert(tau > 0);
   assert(Pi >= 0 && Pi < 1.0);
   // shortcut for no-insertion case
   if (Pi == 0)
      return 0;
   // main computation
   int I = int(ceil((log(Pr) - log(double(tau))) / log(Pi))) - 1;
   I = std::max(I, 1);
#if DEBUG>=3
   std::cerr << "DEBUG (qids): for N = " << tau << ", I = " << I;
#endif
   if (Icap > 0)
      {
      I = std::min(I, Icap);
#if DEBUG>=3
      std::cerr << ", capped to " << I;
#endif
      }
#if DEBUG>=3
   std::cerr << "." << std::endl;
#endif
   return I;
   }

/*!
 * \brief Determine maximum drift at the end of a frame of tau symbols (Davey's algorithm)
 *
 * \f[ x_{max} = Q^{-1}(\frac{P_r}{2}) \sqrt{\frac{\tau p}{1-p}} \f]
 * where \f$ p = P_i = P_d \f$ and \f$ P_r \f$ is an arbitrary probability of
 * having a block of size \f$ \tau \f$ where the drift at the end is greater
 * than \f$ \pm x_{max} \f$.
 * In this class, this value is fixed.
 *
 * The calculation is based on the assumption that the end-of-frame drift has
 * a Gaussian distribution with zero mean and standard deviation given by
 * \f$ \sigma = \sqrt{\frac{2 \tau p}{1-p}} \f$.
 */
template <class G, class real>
int qids<G, real>::metric_computer::compute_xmax_davey(int tau, double Pi,
      double Pd, double Pr)
   {
   // sanity checks
   assert(tau > 0);
   validate(Pd, Pi);
   // set constants
   assertalways(Pi == Pd);
   // assumed by this algorithm
   const double p = Pi;
   // determine required multiplier
   const double factor = libbase::Qinv(Pr / 2.0);
#if DEBUG>=4
   std::cerr << "DEBUG (qids): [davey] Q(" << factor << ") = " << libbase::Q(
         factor) << std::endl;
#endif
   // main computation
   const int xmax = int(ceil(factor * sqrt(2 * tau * p / (1 - p))));
   // tell the user what we did and return
   return xmax;
   }

/*!
 * \brief Determine maximum drift at the end of a frame of tau symbols, given the
 * supplied drift pdf at start of transmission.
 */
template <class G, class real>
int qids<G, real>::metric_computer::compute_xmax(int tau, double Pi, double Pd, double Pr,
      const libbase::vector<double>& sof_pdf, const int offset)
   {
   try
      {
      compute_drift_prob_functor f(compute_drift_prob_exact, sof_pdf, offset);
      const int xmax = compute_xmax_with(f, tau, Pi, Pd, Pr);
#if DEBUG>=4
      std::cerr << "DEBUG (qids): [with exact] for N = " << tau << ", xmax = " << xmax << "." << std::endl;
#endif
      return xmax;
      }
   catch (std::exception&)
      {
      compute_drift_prob_functor f(compute_drift_prob_davey, sof_pdf, offset);
      const int xmax = compute_xmax_with(f, tau, Pi, Pd, Pr);
#if DEBUG>=4
      std::cerr << "DEBUG (qids): [with davey] for N = " << tau << ", xmax = " << xmax << "." << std::endl;
#endif
      return xmax;
      }
   }

/*!
 * \brief Determine maximum drift at the end of a frame of tau symbols, given the
 * supplied drift pdf at start of transmission.
 *
 * This method caps the minimum value of xmax, so it is at least equal to I.
 */
template <class G, class real>
int qids<G, real>::metric_computer::compute_xmax(int tau, double Pi, double Pd, double Pr,
      int I, const libbase::vector<double>& sof_pdf, const int offset)
   {
   // sanity checks
   assert(tau > 0);
   validate(Pd, Pi);
   // use the appropriate algorithm
   int xmax = compute_xmax(tau, Pi, Pd, Pr, sof_pdf, offset);
   // cap minimum value
   xmax = std::max(xmax, I);
   // tell the user what we did and return
#if DEBUG>=3
   std::cerr << "DEBUG (qids): [adjusted] for N = " << tau << ", xmax = "
   << xmax << "." << std::endl;
#endif
   return xmax;
   }

/*!
 * \brief Compute receiver coefficient value
 *
 * \param err Flag for last symbol in error \f[ r_\mu \neq t \f]
 * \param mu Number of insertions in received sequence (0 or more)
 * \return Likelihood for the given sequence
 *
 * When the last symbol \f[ r_\mu = t \f]
 * \f[ Rtable(0,\mu) =
 * \left(\frac{P_i}{2}\right)^\mu
 * \left( (1-P_i-P_d) (1-P_s) + \frac{1}{2} P_i P_d \right)
 * , \mu \in (0, \ldots I) \f]
 *
 * When the last symbol \f[ r_\mu \neq t \f]
 * \f[ Rtable(1,\mu) =
 * \left(\frac{P_i}{2}\right)^\mu
 * \left( (1-P_i-P_d) P_s + \frac{1}{2} P_i P_d \right)
 * , \mu \in (0, \ldots I) \f]
 */
template <class G, class real>
real qids<G, real>::metric_computer::compute_Rtable_entry(bool err, int mu,
      double Ps, double Pd, double Pi)
   {
   const double a1 = (1 - Pi - Pd);
   const double a2 = 0.5 * Pi * Pd;
   const double a3 = pow(0.5 * Pi, mu);
   const double a4 = err ? Ps : (1 - Ps);
   return real(a3 * (a1 * a4 + a2));
   }

/*!
 * \brief Compute receiver coefficient set
 * 
 * First row has elements where the last symbol \f[ r_\mu = t \f]
 * Second row has elements where the last symbol \f[ r_\mu \neq t \f]
 */
template <class G, class real>
void qids<G, real>::metric_computer::compute_Rtable(array2r_t& Rtable, int I,
      double Ps, double Pd, double Pi)
   {
   // Allocate required size
   Rtable.init(2, I + 1);
   // Set values for insertions
   for (int mu = 0; mu <= I; mu++)
      {
      Rtable(0, mu) = compute_Rtable_entry(0, mu, Ps, Pd, Pi);
      Rtable(1, mu) = compute_Rtable_entry(1, mu, Ps, Pd, Pi);
      }
   }

// Internal functions

/*!
 * \brief Sets up pre-computed values
 * 
 * This function computes all cached quantities used within actual channel
 * operations. Since these values depend on the channel conditions, this
 * function should be called any time a channel parameter is changed.
 */
template <class G, class real>
void qids<G, real>::metric_computer::precompute(double Ps, double Pd, double Pi, double Pr,
      int Icap)
   {
   if (N == 0)
      {
      I = 0;
      xmax = 0;
      // reset array
      Rtable.init(0, 0);
      return;
      }
   assert(N > 0);
   // fba decoder parameters
   I = compute_I(N, Pi, Pr, Icap);
   xmax = compute_xmax(N, Pi, Pd, Pr, I);
   // receiver coefficients
   Rval = real(Pd);
#ifdef USE_CUDA
   // create local table and copy to device
   array2r_t Rtable_temp;
   compute_Rtable(Rtable_temp, I, Ps, Pd, Pi);
   Rtable = Rtable_temp;
#else
   compute_Rtable(Rtable, I, Ps, Pd, Pi);
#endif
   // lattice coefficients
   Pval_d = real(Pd);
   Pval_i = real(0.5 * Pi);
   Pval_tc = real((1 - Pi - Pd) * (1 - Ps));
   Pval_te = real((1 - Pi - Pd) * Ps);
   }

/*!
 * \brief Initialization
 * 
 * Sets the block size to an unusable value.
 */
template <class G, class real>
void qids<G, real>::metric_computer::init()
   {
   // set block size to unusable value
   N = 0;
#ifdef USE_CUDA
   // Initialize CUDA
   cuda::cudaInitialize(std::cerr);
#endif
   }

// Channel receiver for host

#ifndef USE_CUDA

// Batch receiver interface - trellis computation
template <class G, class real>
void qids<G, real>::metric_computer::receive_trellis(const array1g_t& tx,
      const array1g_t& rx, array1r_t& ptable) const
   {
   using std::min;
   using std::max;
   using std::swap;
   // Compute sizes
   const int n = tx.size();
   const int rho = rx.size();
   assert(n <= N);
   assert(labs(rho - n) <= xmax);
   // Set up two slices of forward matrix, and associated pointers
   // Arrays are allocated on the stack as a fixed size; this avoids dynamic
   // allocation (which would otherwise be necessary as the size is non-const)
   assertalways(2 * xmax + 1 <= arraysize);
   real F0[arraysize];
   real F1[arraysize];
   real *Fthis = F1 + xmax; // offset by 'xmax' elements
   real *Fprev = F0 + xmax; // offset by 'xmax' elements
   // initialize for j=0
   // for prior list, reset all elements to zero
   for (int x = -xmax; x <= xmax; x++)
      {
      Fthis[x] = 0;
      }
   // we also know x[0] = 0; ie. drift before transmitting symbol t0 is zero.
   Fthis[0] = 1;
   // compute remaining matrix values
   for (int j = 1; j <= n; ++j)
      {
      // swap 'this' and 'prior' lists
      swap(Fthis, Fprev);
      // for this list, reset all elements to zero
      for (int x = -xmax; x <= xmax; x++)
         {
         Fthis[x] = 0;
         }
      // event must fit the received sequence:
      // 1. j-1+a >= 0
      // 2. j-1+y < rx.size()
      // limits on insertions and deletions must be respected:
      // 3. y-a <= I
      // 4. y-a >= -1
      const int ymin = max(-xmax, -j);
      const int ymax = min(xmax, rho - j);
      for (int y = ymin; y <= ymax; ++y)
         {
         real result = 0;
         const int amin = max(max(-xmax, 1 - j), y - I);
         const int amax = min(xmax, y + 1);
         // check if the last element is a pure deletion
         int amax_act = amax;
         if (y - amax < 0)
            {
            real temp = Fprev[amax];
            temp *= Rval;
            result += temp;
            amax_act--;
            }
         // elements requiring comparison of tx and rx symbols
         for (int a = amin; a <= amax_act; ++a)
            {
            // received subsequence has
            // start:  j-1+a
            // length: y-a+1
            // therefore last element is: start+length-1 = j+y-1
            const bool cmp = tx(j - 1) != rx(j + y - 1);
            real temp = Fprev[a];
            temp *= Rtable(cmp, y - a);
            result += temp;
            }
         Fthis[y] = result;
         }
      }
   // copy results and return
   assertalways(ptable.size() == 2 * xmax + 1);
   for (int x = -xmax; x <= xmax; x++)
      {
      ptable(xmax + x) = Fthis[x];
      }
   }

// Batch receiver interface - lattice computation
template <class G, class real>
void qids<G, real>::metric_computer::receive_lattice(const array1g_t& tx,
      const array1g_t& rx, array1r_t& ptable) const
   {
   using std::swap;
   // Compute sizes
   const int n = tx.size();
   const int rho = rx.size();
   // Set up two slices of lattice, and associated pointers
   // Arrays are allocated on the stack as a fixed size; this avoids dynamic
   // allocation (which would otherwise be necessary as the size is non-const)
   assertalways(rho + 1 <= arraysize);
   real F0[arraysize];
   real F1[arraysize];
   real *Fthis = F1;
   real *Fprev = F0;
   // initialize for i=0 (first row of lattice)
   Fthis[0] = 1;
   for (int j = 1; j <= rho; j++)
      Fthis[j] = Fthis[j - 1] * Pval_i;
   // compute remaining rows, except last
   for (int i = 1; i < n; i++)
      {
      // swap 'this' and 'prior' rows
      swap(Fthis, Fprev);
      // handle first column as a special case
      real temp = Fprev[0];
      temp *= Pval_d;
      Fthis[0] = temp;
      // remaining columns
      for (int j = 1; j <= rho; j++)
         {
         const real pi = Fthis[j - 1] * Pval_i;
         const real pd = Fprev[j] * Pval_d;
         const bool cmp = tx(i - 1) == rx(j - 1);
         const real ps = Fprev[j - 1] * (cmp ? Pval_tc : Pval_te);
         real temp = ps + pd;
         temp += pi;
         Fthis[j] = temp;
         }
      }
   // compute last row as a special case (no insertions)
   // swap 'this' and 'prior' rows
   swap(Fthis, Fprev);
   // handle first column as a special case
   real temp = Fprev[0];
   temp *= Pval_d;
   Fthis[0] = temp;
   // remaining columns
   for (int j = 1; j <= rho; j++)
      {
      const real pd = Fprev[j] * Pval_d;
      const bool cmp = tx(n - 1) == rx(j - 1);
      const real ps = Fprev[j - 1] * (cmp ? Pval_tc : Pval_te);
      real temp = ps + pd;
      Fthis[j] = temp;
      }
   // copy results and return
   assertalways(ptable.size() == 2 * xmax + 1);
   for (int x = -xmax; x <= xmax; x++)
      {
      // convert index
      const int j = x + n;
      if (j >= 0 && j <= rho)
         ptable(xmax + x) = Fthis[j];
      else
         ptable(xmax + x) = 0;
      }
   }

// Batch receiver interface - lattice corridor computation
template <class G, class real>
void qids<G, real>::metric_computer::receive_lattice_corridor(
      const array1g_t& tx, const array1g_t& rx,
      array1r_t& ptable) const
   {
   using std::swap;
   using std::min;
   using std::max;
   // Compute sizes
   const int n = tx.size();
   const int rho = rx.size();
   // Set up single slice of lattice on the stack as a fixed size;
   // this avoids dynamic allocation (which would otherwise be necessary
   // as the size is non-const)
   assertalways(rho + 1 <= arraysize);
   real F[arraysize];
   // set up variable to keep track of Fprev[j-1]
   real Fprev;
   // initialize for i=0 (first row of lattice)
   // Fthis[0] = 1;
   F[0] = 1;
   const int jmax = min(xmax, rho);
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
      if (i - xmax <= 0)
         {
         // Fthis[0] = Fprev[0] * Pval_d;
         F[0] = Fprev * Pval_d;
         }
      // determine limits for remaining columns (after first)
      const int jmin = max(i - xmax, 1);
      const int jmax = min(i + xmax, rho);
      // keep Fprev[jmin - 1], if necessary
      if (jmin > 1)
         {
         Fprev = F[jmin - 1];
         }
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
         if (j < i + xmax)
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
   if (i - xmax <= 0)
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
      if (j < i + xmax)
         // temp += Fprev[j] * Pval_d;
         temp += Fprev * Pval_d;
      // store result
      // Fthis[j] = temp;
      F[j] = temp;
      }
   // copy results and return
   assertalways(ptable.size() == 2 * xmax + 1);
   for (int x = -xmax; x <= xmax; x++)
      {
      // convert index
      const int j = x + n;
      if (j >= 0 && j <= rho)
         ptable(xmax + x) = F[j];
      else
         ptable(xmax + x) = 0;
      }
   }

#endif

/*!
 * \brief Initialization
 *
 * Sets the channel with fixed values for Ps, Pd, Pi. This way, if the user
 * never calls set_parameter(), the values are valid.
 */
template <class G, class real>
void qids<G, real>::init()
   {
   // channel parameters
   Ps = fixedPs;
   Pd = fixedPd;
   Pi = fixedPi;
   // drift exclusion probability
   Pr = 1e-10;
   // initialize metric computer
   computer.init();
   computer.precompute(Ps, Pd, Pi, Pr, Icap);
   }

/*!
 * \brief Resize drift pdf table
 *
 * The input pdf table can be any size, with any offset; the data is copied
 * into the output pdf table, going from -xmax to +xmax.
 */
template <class G, class real>
libbase::vector<double> qids<G, real>::resize_drift(const array1d_t& in,
      const int offset, const int xmax)
   {
   // allocate space an initialize
   array1d_t out(2 * xmax + 1);
   out = 0;
   // copy over common elements
   const int imin = std::max(-offset, -xmax);
   const int imax = std::min(in.size() - 1 - offset, xmax);
   const int length = imax - imin + 1;
#if DEBUG>=3
   std::cerr << "DEBUG (qids): [resize] offset = " << offset << ", xmax = "
   << xmax << "." << std::endl;
   std::cerr << "DEBUG (qids): [resize] imin = " << imin << ", imax = "
   << imax << "." << std::endl;
#endif
   out.segment(xmax + imin, length) = in.extract(offset + imin, length);
   // return results
   return out;
   }

// Channel parameter handling

/*!
 * \brief Set channel parameter
 * 
 * This function sets any of Ps, Pd, or Pi that are flagged to change. Any of
 * these parameters that are not flagged to change will instead be set to the
 * specified fixed value.
 *
 * \note We set fixed values every time to ensure that there is no leakage
 * between successive uses of this class. (i.e. once this function is called,
 * the class will be in a known determined state).
 */
template <class G, class real>
void qids<G, real>::set_parameter(const double p)
   {
   const double q = field_utils<G>::elements();
   assertalways(p >=0 && p <= (q-1)/q);
   set_ps(varyPs ? p : fixedPs);
   set_pd(varyPd ? p : fixedPd);
   set_pi(varyPi ? p : fixedPi);
   libbase::trace << "DEBUG (qids): Ps = " << Ps << ", Pd = " << Pd << ", Pi = "
         << Pi << std::endl;
   }

/*!
 * \brief Get channel parameter
 * 
 * This returns the value of the first of Ps, Pd, or Pi that are flagged to
 * change. If none of these are flagged to change, this constitutes an error
 * condition.
 */
template <class G, class real>
double qids<G, real>::get_parameter() const
   {
   assert(varyPs || varyPd || varyPi);
   if (varyPs)
      return Ps;
   if (varyPd)
      return Pd;
   // must be varyPi
   return Pi;
   }

// Channel function overrides

/*!
 * \copydoc channel::corrupt()
 * 
 * \note Due to limitations of the interface, which was designed for
 * substitution channels, only the substitution part of the channel model is
 * handled here.
 * 
 * For symbols that are substituted, any of the remaining symbols are equally
 * likely.
 */
template <class G, class real>
G qids<G, real>::corrupt(const G& s)
   {
   const double p = this->r.fval_closed();
   if (p < Ps)
      return field_utils<G>::corrupt(s, this->r);
   return s;
   }

// Stream-oriented channel characteristics

/*!
 * \brief Get the expected drift distribution after transmitting 'tau' symbols,
 * assuming the start-of-frame drift is zero.
 *
 * This method determines the required limit on state space, and computes the
 * end-of-frame distribution for this range. It returns the necessary offset
 * accordingly.
 */
template <class G, class real>
void qids<G, real>::get_drift_pdf(int tau, double Pr, libbase::vector<double>& eof_pdf,
      libbase::size_type<libbase::vector>& offset) const
   {
   // determine the range of drifts we're interested in
   const int xmax = compute_xmax(tau, Pr);
   // store the necessary offset
   offset = libbase::size_type<libbase::vector>(xmax);
   // initialize result vector
   eof_pdf.init(2 * xmax + 1);
   // compute the probability at each possible drift
   try
      {
      for (int x = -xmax; x <= xmax; x++)
         {
         eof_pdf(x + xmax) = metric_computer::compute_drift_prob_exact(x, tau,
               Pi, Pd);
         }
      }
   catch (std::exception&)
      {
      for (int x = -xmax; x <= xmax; x++)
         {
         eof_pdf(x + xmax) = metric_computer::compute_drift_prob_davey(x, tau,
               Pi, Pd);
         }
      }
   }

/*!
 * \brief Get the expected drift distribution after transmitting 'tau' symbols,
 * assuming the start-of-frame distribution is as given.
 *
 * This method determines an updated limit on state space, and computes the
 * end-of-frame distribution for this range. It also resizes the start-of-frame
 * pdf accordingly and updates the given offset.
 */
template <class G, class real>
void qids<G, real>::get_drift_pdf(int tau, double Pr, libbase::vector<double>& sof_pdf,
      libbase::vector<double>& eof_pdf,
      libbase::size_type<libbase::vector>& offset) const
   {
   // determine the range of drifts we're interested in
   const int xmax = compute_xmax(tau, Pr, sof_pdf, offset);
   // initialize result vector
   eof_pdf.init(2 * xmax + 1);
   // compute the probability at each possible drift
   try
      {
      for (int x = -xmax; x <= xmax; x++)
         {
         eof_pdf(x + xmax) = metric_computer::compute_drift_prob_with(
               metric_computer::compute_drift_prob_exact, x, tau, Pi, Pd,
               sof_pdf, offset);
         }
      }
   catch (std::exception&)
      {
      for (int x = -xmax; x <= xmax; x++)
         {
         eof_pdf(x + xmax) = metric_computer::compute_drift_prob_with(
               metric_computer::compute_drift_prob_davey, x, tau, Pi, Pd,
               sof_pdf, offset);
         }
      }
   // resize start-of-frame pdf
   sof_pdf = resize_drift(sof_pdf, offset, xmax);
   // update with the new offset
   offset = libbase::size_type<libbase::vector>(xmax);
   }

// Channel functions

/*!
 * \copydoc channel::transmit()
 * 
 * The channel model implemented is described by the following state diagram:
 * \dot
 * digraph states {
 * // Make figure left-to-right
 * rankdir = LR;
 * // state definitions
 * this [ shape=circle, color=gray, style=filled, label="t(i)" ];
 * next [ shape=circle, color=gray, style=filled, label="t(i+1)" ];
 * // path definitions
 * this -> Insert [ label="Pi" ];
 * Insert -> this;
 * this -> Delete [ label="Pd" ];
 * Delete -> next;
 * this -> Transmit [ label="1-Pi-Pd" ];
 * Transmit -> next [ label="1-Ps" ];
 * Transmit -> Substitute [ label="Ps" ];
 * Substitute -> next;
 * }
 * \enddot
 * 
 * \note We have initially no idea how long the received sequence will be, so
 * we first determine the state sequence at every timestep, keeping
 * track of:
 * - the number of insertions \e before given position, and
 * - whether the given position is transmitted or deleted.
 * 
 * \note We have to make sure that we don't corrupt the vector we're reading
 * from (in the case where tx and rx are the same vector); therefore,
 * the result is first created as a new vector and only copied over at
 * the end.
 * 
 * \sa corrupt()
 */
template <class G, class real>
void qids<G, real>::transmit(const array1g_t& tx, array1g_t& rx)
   {
   const int tau = tx.size();
   state_ins.init(tau);
   state_ins = 0;
   state_tx.init(tau);
   state_tx = true;
   // determine state sequence
   for (int i = 0; i < tau; i++)
      {
      double p;
      while ((p = this->r.fval_closed()) < Pi)
         state_ins(i)++;
      if (p < (Pi + Pd))
         state_tx(i) = false;
      }
   // Initialize results vector
#if DEBUG>=2
   libbase::trace << "DEBUG (qids): transmit = " << state_tx << std::endl;
   libbase::trace << "DEBUG (qids): insertions = " << state_ins << std::endl;
#endif
   array1g_t newrx;
   newrx.init(tau + get_drift(tau));
   // Corrupt the modulation symbols (simulate the channel)
   for (int i = 0, j = 0; i < tau; i++)
      {
      for (int ins = 0; ins < state_ins(i); ins++)
         newrx(j++) = (this->r.fval_closed() < 0.5);
      if (state_tx(i))
         newrx(j++) = corrupt(tx(i));
      }
   // copy results back
   rx = newrx;
   }

// description output

template <class G, class real>
std::string qids<G, real>::description() const
   {
   std::ostringstream sout;
   sout << field_utils<G>::elements() << "-ary IDS channel (";
   // List varying components
   if (varyPs)
      sout << "Ps=";
   if (varyPi)
      sout << "Pi=";
   if (varyPd)
      sout << "Pd=";
   sout << "p";
   // List non-varying components, with their value
   if (!varyPs)
      sout << ", Ps=" << fixedPs;
   if (!varyPd)
      sout << ", Pd=" << fixedPd;
   if (!varyPi)
      sout << ", Pi=" << fixedPi;
   switch (computer.receiver_type)
      {
      case receiver_trellis:
         sout << ", trellis computation";
         break;
      case receiver_lattice:
         sout << ", lattice computation";
         break;
      case receiver_lattice_corridor:
         sout << ", lattice-corridor computation";
         break;
      default:
         failwith("Unknown receiver mode");
         break;
      }
   sout << ")";
#ifdef USE_CUDA
   sout << " [CUDA]";
#endif
   return sout.str();
   }

// object serialization - saving

template <class G, class real>
std::ostream& qids<G, real>::serialize(std::ostream& sout) const
   {
   sout << "# Version" << std::endl;
   sout << 3 << std::endl;
   sout << "# Vary Ps?" << std::endl;
   sout << varyPs << std::endl;
   sout << "# Vary Pd?" << std::endl;
   sout << varyPd << std::endl;
   sout << "# Vary Pi?" << std::endl;
   sout << varyPi << std::endl;
   sout << "# Cap on I (0=uncapped)" << std::endl;
   sout << Icap << std::endl;
   sout << "# Fixed Ps value" << std::endl;
   sout << fixedPs << std::endl;
   sout << "# Fixed Pd value" << std::endl;
   sout << fixedPd << std::endl;
   sout << "# Fixed Pi value" << std::endl;
   sout << fixedPi << std::endl;
   sout << "# Mode for receiver (0=trellis, 1=lattice, 2=lattice corridor)" << std::endl;
   sout << computer.receiver_type << std::endl;
   return sout;
   }

// object serialization - loading

/*!
 * \version 1 Initial version (based on bsid v.4, without biased flag)
 *
 * \version 2 Added mode for receiver (trellis or lattice)
 *
 * \version 3 Added support for corridor-lattice receiver
 */
template <class G, class real>
std::istream& qids<G, real>::serialize(std::istream& sin)
   {
   // get format version
   int version;
   sin >> libbase::eatcomments >> version >> libbase::verify;
   assertalways(version >= 1);
   // read flags
   sin >> libbase::eatcomments >> varyPs >> libbase::verify;
   sin >> libbase::eatcomments >> varyPd >> libbase::verify;
   sin >> libbase::eatcomments >> varyPi >> libbase::verify;
   // read cap on insertions
   sin >> libbase::eatcomments >> Icap >> libbase::verify;
   // read fixed Ps,Pd,Pi values
   sin >> libbase::eatcomments >> fixedPs >> libbase::verify;
   sin >> libbase::eatcomments >> fixedPd >> libbase::verify;
   sin >> libbase::eatcomments >> fixedPi >> libbase::verify;
   // read receiver mode if present
   if (version >= 2)
      {
      int temp;
      sin >> libbase::eatcomments >> temp >> libbase::verify;
      computer.receiver_type = (receiver_t) temp;
      }
   else
      computer.receiver_type = receiver_trellis;
   // initialise the object and return
   init();
   return sin;
   }

} // end namespace

#include "gf.h"
#include "mpgnu.h"
#include "logrealfast.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::mpgnu;
using libbase::logrealfast;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define SYMBOL_TYPE_SEQ \
   (bool) \
   GF_TYPE_SEQ
#ifdef USE_CUDA
#define REAL_TYPE_SEQ \
   (float)(double)
#else
#define REAL_TYPE_SEQ \
   (float)(double)(mpgnu)(logrealfast)
#endif

/* Serialization string: qids<type,real>
 * where:
 *      type = bool | gf2 | gf4 ...
 *      real = float | double | [mpgnu | logrealfast (CPU only)]
 */
#define INSTANTIATE(r, args) \
      template class qids<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer qids<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "channel", \
            "qids<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) ">", \
            qids<BOOST_PP_SEQ_ENUM(args)>::create);

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (SYMBOL_TYPE_SEQ)(REAL_TYPE_SEQ))

} // end namespace
