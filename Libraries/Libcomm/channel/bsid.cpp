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

#include "bsid.h"
#include "itfunc.h"
#include <sstream>
#include <limits>
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

const libbase::serializer bsid::shelper("channel", "bsid", bsid::create);
const double bsid::metric_computer::Pr = 1e-10;

// FBA decoder parameter computation

/*!
 * \brief The probability of drift x after transmitting tau bits
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
 * integral over x +/- 0.5.
 *
 * Since the probability is symmetric about x=0, we always use positive x;
 * this ensures the subtraction does not fail due to resolution of double.
 */
double bsid::metric_computer::compute_drift_prob_davey(int x, int tau,
      double Pi, double Pd)
   {
   // sanity checks
   assert(tau > 0);
   validate(Pd, Pi);
   // set constants
   assert(Pi == Pd); // assumed by this algorithm
   const double p = Pi;
   // the distribution is approximately Gaussian with:
   const double sigma = sqrt(2 * tau * p / (1 - p));
   // main computation
   const double xa = abs(x);
   const double x1 = (xa - 0.5) / sigma;
   const double x2 = (xa + 0.5) / sigma;
   const double this_p = libbase::Q(x1) - libbase::Q(x2);
#if DEBUG>=4
   std::cerr << "DEBUG (bsid): [pdf-davey] x = " << x << ", sigma = " << sigma
   << ", this_p = " << this_p << "." << std::endl;
#endif
   return this_p;
   }

/*!
 * \brief The probability of drift x after transmitting tau bits
 *
 * Computes the required probability using the exact metric from our
 * Trans. Inf. Theory submission.
 *
 * \todo Add formula
 */
double bsid::metric_computer::compute_drift_prob_exact(int x, int tau,
      double Pi, double Pd)
   {
   typedef long double real;
   // sanity checks
   assert(tau > 0);
   validate(Pd, Pi);
   // set constants
   const double Pt = 1 - Pi - Pd;
   const double Pf = Pi * Pd / Pt;
   const int imin = (x < 0) ? -x : 0;
   // shortcut for out-of-range values
   if (imin > tau)
      return 0;
   // compute initial value
   real p0 = 1;
   // inner and outer factors (do this in log domain)
   p0 = log(p0);
   // inner factor if x > 0 (and therefore imin = 0)
   for (int xi = 1; xi <= x + imin; xi++)
      {
      p0 += log(real(tau + xi - 1)) - log(real(xi));
      }
   // inner factor if x < 0 (and therefore imin > 0)
   if (imin > 0)
      p0 += log(Pf) * imin;
   for (int i = 1; i <= imin; i++)
      {
      p0 += log(real(tau - i + 1)) - log(real(i));
      }
   // outer factors
   p0 += log(real(Pt)) * tau;
   p0 += log(real(Pi)) * x;
   p0 = exp(p0);
#if DEBUG>=4
   std::cerr << "DEBUG (bsid): [pdf-exact] x = " << x << ", p0 = " << p0
   << std::endl;
#endif
   if (p0 == 0)
      throw std::overflow_error("zero factor");
   // main computation
   real this_p = p0;
   for (int i = imin + 1; i <= tau; i++)
      {
      // update factor
      p0 *= Pf;
      p0 *= real(tau + x + i - 1) / real(x + i);
      p0 *= real(tau - i + 1) / real(i);
      // update main result
      const real last_p = this_p;
      this_p += p0;
#if DEBUG>=4
      std::cerr << "DEBUG (bsid): [pdf-exact] i = " << i << ", p0 = " << p0
      << ", this_p = " << this_p << std::endl;
#endif
      // early cutoff
      if (this_p == last_p)
         break;
      }
#if DEBUG>=4
   std::cerr << "DEBUG (bsid): [pdf-exact] this_p = " << this_p << std::endl;
#endif
   // validate and return result
   if (!std::isfinite(this_p))
      throw std::overflow_error("value not finite");
   else if (this_p == 0)
      throw std::overflow_error("zero value");
   else if (this_p < 0)
      throw std::overflow_error("negative value");
   return this_p;
   }

/*!
 * \brief The probability of drift x after transmitting tau bits, given the
 * supplied drift pdf at start of transmission.
 *
 * The final drift pdf is obtained by convolving the expected pdf for a
 * known start of frame position with the actual start of frame distribution.
 */
template <typename F>
double bsid::metric_computer::compute_drift_prob_with(const F& compute_pdf,
      int x, int tau, double Pi, double Pd,
      const libbase::vector<double>& sof_pdf, const int offset)
   {
   // if sof_pdf is empty, delegate automatically
   if (sof_pdf.size() == 0)
      return compute_pdf(x, tau, Pi, Pd);
   // compute the probability at requested drift
   double this_p = 0;
   const int imin = -offset;
   const int imax = sof_pdf.size() - offset;
   for (int i = imin; i < imax; i++)
      {
      const double p = compute_pdf(x - i, tau, Pi, Pd);
      this_p += sof_pdf(i + offset) * p;
      }
   // normalize
   this_p /= sof_pdf.sum();
   // confirm that this value is finite and valid
   assert(this_p >= 0 && this_p < std::numeric_limits<double>::infinity());
   return this_p;
   }

/*!
 * \brief Determine limit for insertions between two time-steps
 * 
 * \f[ I = \left\lceil \frac{ \log{P_r} - \log \tau }{ \log P_i } \right\rceil - 1 \f]
 * where \f$ P_r \f$ is an arbitrary probability of having a block of size
 * \f$ \tau \f$ with at least one event of more than \f$ I \f$ insertions
 * between successive time-steps.
 * In this class, this value is fixed at \f$ P_r = 10^{-12} \f$.
 * 
 * \note The smallest allowed value is \f$ I = 1 \f$; the largest value depends
 * on a user parameter.
 */
int bsid::metric_computer::compute_I(int tau, double Pi, int Icap)
   {
   // sanity checks
   assert(tau > 0);
   assert(Pi >= 0 && Pi < 1.0);
   // main computation
   int I = int(ceil((log(Pr) - log(double(tau))) / log(Pi))) - 1;
   I = std::max(I, 1);
#if DEBUG>=3
   std::cerr << "DEBUG (bsid): for N = " << tau << ", I = " << I;
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
 * \brief Determine maximum drift at the end of a frame of tau bits (Davey's algorithm)
 *
 * \f[ x_{max} = Q^{-1}(\frac{P_r}{2}) \sqrt{\frac{\tau p}{1-p}} \f]
 * where \f$ p = P_i = P_d \f$ and \f$ P_r \f$ is an arbitrary probability of
 * having a block of size \f$ \tau \f$ where the drift at the end is greater
 * than \f$ \pm x_{max} \f$.
 * In this class, this value is fixed at \f$ P_r = 10^{-12} \f$.
 *
 * The calculation is based on the assumption that the end-of-frame drift has
 * a Gaussian distribution with zero mean and standard deviation given by
 * \f$ \sigma = \sqrt{\frac{2 \tau p}{1-p}} \f$.
 */
int bsid::metric_computer::compute_xmax_davey(int tau, double Pi, double Pd)
   {
   // sanity checks
   assert(tau > 0);
   validate(Pd, Pi);
   // set constants
   assert(Pi == Pd); // assumed by this algorithm
   const double p = Pi;
   // determine required multiplier
   const double factor = libbase::Qinv(Pr / 2.0);
#if DEBUG>=4
   std::cerr << "DEBUG (bsid): [davey] Q(" << factor << ") = " << libbase::Q(
         factor) << std::endl;
#endif
   // main computation
   const int xmax = int(ceil(factor * sqrt(2 * tau * p / (1 - p))));
   // tell the user what we did and return
   return xmax;
   }

/*!
 * \brief Determine maximum drift at the end of a frame of tau bits, given the
 * supplied drift pdf at start of transmission.
 *
 * The drift range is chosen such that the probability of having the drift
 * after transmitting \f$ \tau \f$ bits being greater than \f$ \pm x_{max} \f$
 * is less than an arbirarty value \f$ P_r \f$.
 * In this class, this value is fixed at \f$ P_r = 10^{-12} \f$.
 */
template <typename F>
int bsid::metric_computer::compute_xmax_with(const F& compute_pdf, int tau,
      double Pi, double Pd)
   {
   // sanity checks
   assert(tau > 0);
   validate(Pd, Pi);
   // determine area that needs to be covered
   double acc = 1.0;
   // determine xmax to use
   int xmax = 0;
   acc -= compute_pdf(xmax, tau, Pi, Pd);
#if DEBUG>=4
   std::cerr << "DEBUG (bsid): xmax = " << xmax << ", acc = " << acc << "."
   << std::endl;
#endif
   while (acc >= Pr)
      {
      xmax++;
      acc -= compute_pdf(xmax, tau, Pi, Pd);
      acc -= compute_pdf(-xmax, tau, Pi, Pd);
#if DEBUG>=4
      std::cerr << "DEBUG (bsid): xmax = " << xmax << ", acc = " << acc << "."
      << std::endl;
#endif
      }
   // tell the user what we did and return
#if DEBUG>=3
   std::cerr << "DEBUG (bsid): [computed] for N = " << tau << ", xmax = "
   << xmax << "." << std::endl;
   std::cerr << "DEBUG (bsid): [davey] for N = " << tau << ", xmax = "
   << compute_xmax_davey(tau, Pi, Pd) << "." << std::endl;
#endif
   return xmax;
   }

// Explicit instantiations

template int bsid::metric_computer::compute_xmax_with(
      const compute_drift_prob_functor& f, int tau, double Pi, double Pd);
template int bsid::metric_computer::compute_xmax_with(
      const compute_drift_prob_functor::pdf_func_t& f, int tau, double Pi,
      double Pd);

/*!
 * \brief Determine maximum drift at the end of a frame of tau bits, given the
 * supplied drift pdf at start of transmission.
 *
 * This method caps the minimum value of xmax, so it is at least equal to I.
 */
int bsid::metric_computer::compute_xmax(int tau, double Pi, double Pd, int I,
      const libbase::vector<double>& sof_pdf, const int offset)
   {
   // sanity checks
   assert(tau > 0);
   validate(Pd, Pi);
   // use the appropriate algorithm
   int xmax = compute_xmax(tau, Pi, Pd, sof_pdf, offset);
   // cap minimum value
   xmax = std::max(xmax, I);
   // tell the user what we did and return
#if DEBUG>=3
   std::cerr << "DEBUG (bsid): [adjusted] for N = " << tau << ", xmax = "
   << xmax << "." << std::endl;
#endif
   return xmax;
   }

/*!
 * \brief Compute receiver coefficient value
 *
 * \param err Flag for last bit in error \f[ r_\mu \neq t \f]
 * \param mu Number of insertions in received sequence (0 or more)
 * \return Likelihood for the given sequence
 *
 * When the last bit \f[ r_\mu = t \f]
 * \f[ Rtable(0,\mu) =
 * \left(\frac{P_i}{2}\right)^\mu
 * \left( (1-P_i-P_d) (1-P_s) + \frac{1}{2} P_i P_d \right)
 * , \mu \in (0, \ldots I) \f]
 *
 * When the last bit \f[ r_\mu \neq t \f]
 * \f[ Rtable(1,\mu) =
 * \left(\frac{P_i}{2}\right)^\mu
 * \left( (1-P_i-P_d) P_s + \frac{1}{2} P_i P_d \right)
 * , \mu \in (0, \ldots I) \f]
 */
bsid::real bsid::metric_computer::compute_Rtable_entry(bool err, int mu,
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
 * First row has elements where the last bit \f[ r_\mu = t \f]
 * Second row has elements where the last bit \f[ r_\mu \neq t \f]
 */
void bsid::metric_computer::compute_Rtable(array2r_t& Rtable, int I, double Ps,
      double Pd, double Pi)
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
void bsid::metric_computer::precompute(double Ps, double Pd, double Pi,
      int Icap, bool biased)
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
   I = compute_I(N, Pi, Icap);
   xmax = compute_xmax(N, Pi, Pd, I);
   // receiver coefficients
   Rval = real(biased ? Pd * Pd : Pd);
#ifdef USE_CUDA
   // create local table and copy to device
   array2r_t Rtable_temp;
   compute_Rtable(Rtable_temp, I, Ps, Pd, Pi);
   Rtable = Rtable_temp;
#else
   compute_Rtable(Rtable, I, Ps, Pd, Pi);
#endif
   }

/*!
 * \brief Initialization
 * 
 * Sets the block size to an unusable value.
 */
void bsid::metric_computer::init()
   {
   // set block size to unusable value
   N = 0;
#ifdef USE_CUDA
   // Initialize CUDA
   cuda::cudaInitialize(std::cerr);
#endif
   }

// Channel receiver for host

bsid::real bsid::metric_computer::receive(const bitfield& tx,
      const array1b_t& rx) const
   {
   // Compute sizes
   const int n = tx.size();
   const int mu = rx.size() - n;
   // Allocate space for results and call main receiver
   static array1r_t ptable;
   ptable.init(2 * xmax + 1);
   receive(tx, rx, ptable);
   // return result
   return ptable(xmax + mu);
   }

#ifndef USE_CUDA
void bsid::metric_computer::receive(const bitfield& tx, const array1b_t& rx,
      array1r_t& ptable) const
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
   real *Fthis = F1;
   real *Fprev = F0;
   // initialize for j=0
   // for prior list, reset all elements to zero
   for (int x = 0; x < 2 * xmax + 1; x++)
      {
      Fthis[x] = 0;
      }
   // we also know x[0] = 0; ie. drift before transmitting bit t0 is zero.
   Fthis[xmax + 0] = 1;
   // compute remaining matrix values
   for (int j = 1; j <= n; ++j)
      {
      // swap 'this' and 'prior' lists
      swap(Fthis, Fprev);
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
      const int ymax = min(2 * xmax, xmax + rho - j);
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
      }
   // copy results and return
   assertalways(ptable.size() == 2 * xmax + 1);
   for (int x = 0; x < 2 * xmax + 1; x++)
      {
      ptable(x) = Fthis[x];
      }
   }
#endif

/*!
 * \brief Initialization
 *
 * Sets the channel with fixed values for Ps, Pd, Pi. This way, if the user
 * never calls set_parameter(), the values are valid.
 */
void bsid::init()
   {
   // channel parameters
   Ps = fixedPs;
   Pd = fixedPd;
   Pi = fixedPi;
   // initialize metric computer
   computer.init();
   computer.precompute(Ps, Pd, Pi, Icap, biased);
   }

/*!
 * \brief Resize drift pdf table
 *
 * The input pdf table can be any size, with any offset; the data is copied
 * into the output pdf table, going from -xmax to +xmax.
 */
libbase::vector<double> bsid::resize_drift(const array1d_t& in,
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
   std::cerr << "DEBUG (bsid): [resize] offset = " << offset << ", xmax = "
   << xmax << "." << std::endl;
   std::cerr << "DEBUG (bsid): [resize] imin = " << imin << ", imax = "
   << imax << "." << std::endl;
#endif
   out.segment(xmax + imin, length) = in.extract(offset + imin, length);
   // return results
   return out;
   }

// Constructors / Destructors

/*!
 * \brief Principal constructor
 * 
 * \sa init()
 */
bsid::bsid(const bool varyPs, const bool varyPd, const bool varyPi,
      const bool biased) :
   biased(biased), varyPs(varyPs), varyPd(varyPd), varyPi(varyPi), Icap(0),
         fixedPs(0), fixedPd(0), fixedPi(0)
   {
   // channel update flags
   assert(varyPs || varyPd || varyPi);
   // other initialization
   init();
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
void bsid::set_parameter(const double p)
   {
   set_ps(varyPs ? p : fixedPs);
   set_pd(varyPd ? p : fixedPd);
   set_pi(varyPi ? p : fixedPi);
   libbase::trace << "DEBUG (bsid): Ps = " << Ps << ", Pd = " << Pd
         << ", Pi = " << Pi << std::endl;
   }

/*!
 * \brief Get channel parameter
 * 
 * This returns the value of the first of Ps, Pd, or Pi that are flagged to
 * change. If none of these are flagged to change, this constitutes an error
 * condition.
 */
double bsid::get_parameter() const
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
 * For the purposes of this channel, a \e substitution corresponds to a symbol
 * inversion.
 */
bool bsid::corrupt(const bool& s)
   {
   const double p = r.fval_closed();
   if (p < Ps)
      return !s;
   return s;
   }

// Stream-oriented channel characteristics

void bsid::get_drift_pdf(int tau, libbase::vector<double>& eof_pdf,
      libbase::size_type<libbase::vector>& offset) const
   {
   // determine the range of drifts we're interested in
   const int xmax = compute_xmax(tau);
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

void bsid::get_drift_pdf(int tau, libbase::vector<double>& sof_pdf,
      libbase::vector<double>& eof_pdf,
      libbase::size_type<libbase::vector>& offset) const
   {
   // determine the range of drifts we're interested in
   const int xmax = compute_xmax(tau, sof_pdf, offset);
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
 * digraph bsidstates {
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
void bsid::transmit(const array1b_t& tx, array1b_t& rx)
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
      while ((p = r.fval_closed()) < Pi)
         state_ins(i)++;
      if (p < (Pi + Pd))
         state_tx(i) = false;
      }
   // Initialize results vector
#if DEBUG>=2
   libbase::trace << "DEBUG (bsid): transmit = " << state_tx << std::endl;
   libbase::trace << "DEBUG (bsid): insertions = " << state_ins << std::endl;
#endif
   array1b_t newrx;
   newrx.init(tau + get_drift(tau));
   // Corrupt the modulation symbols (simulate the channel)
   for (int i = 0, j = 0; i < tau; i++)
      {
      for (int ins = 0; ins < state_ins(i); ins++)
         newrx(j++) = (r.fval_closed() < 0.5);
      if (state_tx(i))
         newrx(j++) = corrupt(tx(i));
      }
   // copy results back
   rx = newrx;
   }

void bsid::receive(const array1b_t& tx, const array1b_t& rx, array1vd_t& ptable) const
   {
   // Compute sizes
   const int M = tx.size();
   // Initialize results vector
   ptable.init(1);
   ptable(0).init(M);
   // Compute results for each possible signal
   for (int x = 0; x < M; x++)
      ptable(0)(x) = bsid::receive(tx(x), rx);
   }

// description output

std::string bsid::description() const
   {
   std::ostringstream sout;
   sout << "BSID channel (";
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
   if (biased)
      sout << ", biased";
   sout << ")";
#ifdef USE_CUDA
   sout << " [CUDA]";
#endif
   return sout.str();
   }

// object serialization - saving

std::ostream& bsid::serialize(std::ostream& sout) const
   {
   sout << "# Version" << std::endl;
   sout << 4 << std::endl;
   sout << "# Biased?" << std::endl;
   sout << biased << std::endl;
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
   return sout;
   }

// object serialization - loading

/*!
 * \version 0 Initial version (un-numbered)
 *
 * \version 1 Added version numbering
 *
 * \version 2 Added 'biased' flag
 *
 * \version 3 Added 'Icap' parameter
 *
 * \version 4 Added fixed Ps,Pd,Pi values
 */
std::istream& bsid::serialize(std::istream& sin)
   {
   std::streampos start = sin.tellg();
   // get format version
   int version;
   sin >> libbase::eatcomments >> version >> libbase::verify;
   // handle old-format files (without version number)
   if (version < 2)
      {
      //sin.clear();
      sin.seekg(start);
      version = 1;
      }
   // read flag if present
   if (version < 2)
      biased = false;
   else
      sin >> libbase::eatcomments >> biased >> libbase::verify;
   sin >> libbase::eatcomments >> varyPs >> libbase::verify;
   sin >> libbase::eatcomments >> varyPd >> libbase::verify;
   sin >> libbase::eatcomments >> varyPi >> libbase::verify;
   // read cap on insertions, if present
   if (version < 3)
      Icap = 2;
   else
      sin >> libbase::eatcomments >> Icap >> libbase::verify;
   // read fixed Ps,Pd,Pi values if present
   if (version < 4)
      {
      fixedPs = 0;
      fixedPd = 0;
      fixedPi = 0;
      }
   else
      {
      sin >> libbase::eatcomments >> fixedPs >> libbase::verify;
      sin >> libbase::eatcomments >> fixedPd >> libbase::verify;
      sin >> libbase::eatcomments >> fixedPi >> libbase::verify;
      }
   // initialise the object and return
   init();
   return sin;
   }

} // end namespace
