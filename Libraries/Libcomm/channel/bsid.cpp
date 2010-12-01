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

/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "bsid.h"
#include "secant.h"
#include "event_timer.h"
#include <sstream>
#include <limits>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show transmit and insertion state vectors during transmission
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

const libbase::serializer bsid::shelper("channel", "bsid", bsid::create);

// FBA decoder parameter computation

/*!
 * \brief Determine limit for insertions between two time-steps
 * 
 * \f[ I = \left\lceil \frac{ \log{P_r} - \log \tau }{ \log p } \right\rceil - 1 \f]
 * where \f$ P_r \f$ is an arbitrary probability of having a block of size
 * \f$ \tau \f$ with at least one event of more than \f$ I \f$ insertions
 * between successive time-steps.
 * In this class, this value is fixed at \f$ P_r = 10^{-12} \f$.
 * 
 * \note The smallest allowed value is \f$ I = 1 \f$; the largest value depends
 * on a user parameter.
 */
int bsid::metric_computer::compute_I(int tau, double p, int Icap)
   {
   int I = int(ceil((log(1e-12) - log(double(tau))) / log(p))) - 1;
   I = std::max(I, 1);
   libbase::trace << "DEBUG (bsid): for N = " << tau << ", I = " << I;
   if (Icap > 0)
      {
      I = std::min(I, Icap);
      libbase::trace << ", capped to " << I;
      }
   libbase::trace << "." << std::endl;
   return I;
   }

/*!
 * \brief Determine maximum drift over a whole N-bit block
 * 
 * \f[ x_{max} = Q^{-1}(\frac{P_r}{2}) \sqrt{\frac{\tau p}{1-p}} \f]
 * where \f$ p = P_i = P_d \f$ and \f$ P_r \f$ is an arbitrary probability of
 * having a block of size \f$ \tau \f$ where the drift at the end is greater
 * than \f$ \pm x_{max} \f$.
 * In this class, this value is fixed at \f$ P_r = 10^{-12} \f$.
 * 
 * The calculation is based on the assumption that the end-of-frame drift has
 * a Gaussian distribution with zero mean and standard deviation given by
 * \f$ \sigma = \sqrt{\frac{\tau p}{1-p}} \f$.
 * 
 * \note The smallest allowed value is \f$ x_{max} = I \f$
 */
int bsid::metric_computer::compute_xmax(int tau, double p, int I)
   {
   // rather than computing the factor using a root-finding method,
   // we fix factor = 7.1305, corresponding to Qinv(1e-12/2.0)
   const double factor = 7.1305;
   int xmax = int(ceil(factor * sqrt(tau * p / (1 - p))));
   xmax = std::max(xmax, I);
   libbase::trace << "DEBUG (bsid): for N = " << tau << ", xmax = " << xmax;
   //xmax = min(xmax,25);
   //libbase::trace << ", capped to " << xmax;
   libbase::trace << "." << std::endl;
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
   xmax = compute_xmax(N, Pi, I);
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

// Channel received for host

#ifndef USE_CUDA
bsid::real bsid::metric_computer::receive(const bitfield& tx,
      const array1b_t& rx) const
   {
   using std::min;
   using std::max;
   using std::swap;
   // Compute sizes
   const int n = tx.size();
   const int mu = rx.size() - n;
   assert(n <= N);
   assert(labs(mu) <= xmax);
   // Set up two slices of forward matrix, and associated pointers
   // Arrays are allocated on the stack as a fixed size; this avoids dynamic
   // allocation (which would otherwise be necessary as the size is non-const)
   const int arraysize = 2 * 15 + 1;
   assert(2 * xmax + 1 <= arraysize);
   real F0[arraysize];
   real F1[arraysize];
   real *Fthis = F1;
   real *Fprev = F0;
   // for prior list, reset all elements to zero
   for (int x = 0; x < 2 * xmax + 1; x++)
      Fprev[x] = 0;
   // we also know x[0] = 0; ie. drift before transmitting bit t0 is zero.
   Fprev[xmax + 0] = 1;
   // compute remaining matrix values
   for (int j = 1; j < n; ++j)
      {
      // for this list, reset all elements to zero
      for (int x = 0; x < 2 * xmax + 1; x++)
         Fthis[x] = 0;
      // event must fit the received sequence:
      // 1. j-1+a >= 0
      // 2. j-1+y < rx.size()
      // limits on insertions and deletions must be respected:
      // 3. y-a <= I
      // 4. y-a >= -1
      // note: a and y are offset by xmax
      const int ymin = max(0, xmax - j);
      const int ymax = min(2 * xmax, xmax + rx.size() - j);
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
      // swap 'this' and 'prior' lists
      swap(Fthis, Fprev);
      }
   // Compute forward metric for known drift, and return
   real result = 0;
   // event must fit the received sequence:
   // 1. n-1+a >= 0
   // 2. n-1+mu < rx.size() [automatically satisfied by definition of mu]
   // limits on insertions and deletions must be respected:
   // 3. mu-a <= I
   // 4. mu-a >= -1
   // note: muoff and a are offset by xmax
   const int muoff = mu + xmax;
   const int amin = max(max(0, muoff - I), xmax + 1 - n);
   const int amax = min(2 * xmax, muoff + 1);
   // check if the last element is a pure deletion
   int amax_act = amax;
   if (muoff - amax < 0)
      {
      result += Fprev[amax] * Rval;
      amax_act--;
      }
   // elements requiring comparison of tx and rx bits
   for (int a = amin; a <= amax_act; ++a)
      {
      // received subsequence has
      // start:  n-1+a
      // length: mu-a+1
      // therefore last element is: start+length-1 = n+mu-1
      const bool cmp = tx(n - 1) != rx(n + mu - 1);
      result += Fprev[a] * Rtable(cmp, muoff - a);
      }
   // clean up and return
   return result;
   }
#endif

/*!
 * \brief Initialization
 *
 * Sets the channel with \f$ P_s = P_d = P_i = 0 \f$. This way, any of the
 * parameters not flagged to change with channel parameter will remain zero.
 */
void bsid::init()
   {
   // channel parameters
   Ps = 0;
   Pd = 0;
   Pi = 0;
   // initialize metric computer
   computer.init();
   computer.precompute(Ps, Pd, Pi, Icap, biased);
   }

// Constructors / Destructors

/*!
 * \brief Principal constructor
 * 
 * \sa init()
 */
bsid::bsid(const bool varyPs, const bool varyPd, const bool varyPi,
      const bool biased) :
   biased(biased), varyPs(varyPs), varyPd(varyPd), varyPi(varyPi), Icap(2)
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
 * This function sets any of Ps, Pd, or Pi that are flagged to change. Any of these
 * parameters that are not flagged to change will instead be set to zero. This ensures
 * that there is no leakage between successive uses of this class. (i.e. once this
 * function is called, the class will be in a known determined state).
 */
void bsid::set_parameter(const double p)
   {
   set_ps(varyPs ? p : 0);
   set_pd(varyPd ? p : 0);
   set_pi(varyPi ? p : 0);
   libbase::trace << "DEBUG (bsid): Ps = " << Ps << ", Pd = " << Pd
         << ", Pi = " << Pi << std::endl;
   }

/*!
 * \brief Get channel parameter
 * 
 * This returns the value of the first of Ps, Pd, or Pi that are flagged to change.
 * If none of these are flagged to change, this constitutes an error condition.
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
 * \note Due to limitations of the interface, which was designed for substitution channels,
 * only the substitution part of the channel model is handled here.
 * 
 * For the purposes of this channel, a \e substitution corresponds to a symbol inversion.
 * This corresponds to the \f$ 0 \Leftrightarrow 1 \f$ binary substitution when used with BPSK
 * modulation. For MPSK modulation, this causes the output to be the symbol farthest away
 * from the input.
 */
bool bsid::corrupt(const bool& s)
   {
   const double p = r.fval_closed();
   if (p < Ps)
      return !s;
   return s;
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
   libbase::vector<int> insertions(tau);
   insertions = 0;
   libbase::vector<int> transmit(tau);
   transmit = 1;
   // determine state sequence
   for (int i = 0; i < tau; i++)
      {
      double p;
      while ((p = r.fval_closed()) < Pi)
         insertions(i)++;
      if (p < (Pi + Pd))
         transmit(i) = 0;
      }
   // Initialize results vector
#if DEBUG>=2
   libbase::trace << "DEBUG (bsid): transmit = " << transmit << std::endl;
   libbase::trace << "DEBUG (bsid): insertions = " << insertions << std::endl;
#endif
   array1b_t newrx;
   newrx.init(transmit.sum() + insertions.sum());
   // Corrupt the modulation symbols (simulate the channel)
   for (int i = 0, j = 0; i < tau; i++)
      {
      while (insertions(i)--)
         newrx(j++) = (r.fval_closed() < 0.5);
      if (transmit(i))
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
   sout << "BSID channel (" << varyPs << varyPd << varyPi;
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
   sout << 3 << std::endl;
   sout << biased << std::endl;
   sout << varyPs << std::endl;
   sout << varyPd << std::endl;
   sout << varyPi << std::endl;
   sout << Icap << std::endl;
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
 */
std::istream& bsid::serialize(std::istream& sin)
   {
   std::streampos start = sin.tellg();
   // get format version
   int version;
   sin >> libbase::eatcomments >> version;
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
      sin >> libbase::eatcomments >> biased;
   sin >> libbase::eatcomments >> varyPs;
   sin >> libbase::eatcomments >> varyPd;
   sin >> libbase::eatcomments >> varyPi;
   // read cap on insertions, if present
   if (version < 3)
      Icap = 2;
   else
      sin >> libbase::eatcomments >> Icap;
   init();
   return sin;
   }

} // end namespace
