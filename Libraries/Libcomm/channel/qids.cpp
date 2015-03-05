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

#include "qids.h"
#include <sstream>
#include <exception>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show transmit and insertion state vectors during transmission process
// 3 - Show details of pdf resizing
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

// FBA decoder parameter computation

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
 * , \mu \in (0, \ldots m1_max) \f]
 *
 * When the last symbol \f[ r_\mu \neq t \f]
 * \f[ Rtable(1,\mu) =
 * \left(\frac{P_i}{2}\right)^\mu
 * \left( (1-P_i-P_d) P_s + \frac{1}{2} P_i P_d \right)
 * , \mu \in (0, \ldots m1_max) \f]
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
void qids<G, real>::metric_computer::compute_Rtable(array2r_t& Rtable, int m1_max,
      double Ps, double Pd, double Pi)
   {
   // Allocate required size
   Rtable.init(2, m1_max + 1);
   // Set values for insertions
   for (int mu = 0; mu <= m1_max; mu++)
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
void qids<G, real>::metric_computer::precompute(double Ps, double Pd, double Pi,
      int T, int mT_min, int mT_max, int m1_min, int m1_max)
   {
   // block size
   this->T = T;
   // fba decoder parameters
   this->mT_min = mT_min;
   this->mT_max = mT_max;
   this->m1_min = m1_min;
   this->m1_max = m1_max;
   // receiver coefficients
   Rval = real(Pd);
#ifdef USE_CUDA
   // create local table and copy to device
   array2r_t Rtable_temp;
   compute_Rtable(Rtable_temp, m1_max, Ps, Pd, Pi);
   Rtable = Rtable_temp;
#else
   compute_Rtable(Rtable, m1_max, Ps, Pd, Pi);
#endif
   // lattice coefficients
   Pval_d = real(Pd);
   Pval_i = real(0.5 * Pi);
   Pval_tc = real((1 - Pi - Pd) * (1 - Ps));
   Pval_te = real((1 - Pi - Pd) * Ps);
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
   //assert(n <= T);
   assert(rho - n <= mT_max);
   assert(rho - n >= mT_min);
   // Set up two slices of forward matrix, and associated pointers
   // Arrays are allocated on the stack as a fixed size; this avoids dynamic
   // allocation (which would otherwise be necessary as the size is non-const)
   assertalways(mT_max - mT_min + 1 <= arraysize);
   real F0[arraysize];
   real F1[arraysize];
   real *Fthis = F1 - mT_min; // offset: first element is index 'mT_min'
   real *Fprev = F0 - mT_min; // offset: first element is index 'mT_min'
   // initialize for j=0
   // for prior list, reset all elements to zero
   for (int x = mT_min; x <= mT_max; x++)
      Fthis[x] = 0;
   // we also know x[0] = 0; ie. drift before transmitting symbol t0 is zero.
   Fthis[0] = 1;
   // compute remaining matrix values
   for (int j = 1; j <= n; ++j)
      {
      // swap 'this' and 'prior' lists
      swap(Fthis, Fprev);
      // for this list, reset all elements to zero
      for (int x = mT_min; x <= mT_max; x++)
         Fthis[x] = 0;
      // event must fit the received sequence:
      // 1. j-1+a >= 0
      // 2. j-1+y < rx.size()
      // limits on insertions and deletions must be respected:
      // 3. y-a <= m1_max
      // 4. y-a >= m1_min
      const int ymin = max(mT_min, -j);
      const int ymax = min(mT_max, rho - j);
      for (int y = ymin; y <= ymax; ++y)
         {
         real result = 0;
         const int amin = max(max(mT_min, 1 - j), y - m1_max);
         const int amax = min(mT_max, y - m1_min);
         // check if the last element is a pure deletion
         int amax_act = amax;
         if (y - amax < 0)
            {
            result += Fprev[amax] * Rval;
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
            result += Fprev[a] * Rtable(cmp, y - a);
            }
         Fthis[y] = result;
         }
      }
   // copy results and return
   assertalways(ptable.size() == mT_max - mT_min + 1);
   for (int x = mT_min; x <= mT_max; x++)
      ptable(x - mT_min) = Fthis[x];
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
   assertalways(ptable.size() == mT_max - mT_min + 1);
   for (int x = mT_min; x <= mT_max; x++)
      {
      // convert index
      const int j = x + n;
      if (j >= 0 && j <= rho)
         ptable(x - mT_min) = Fthis[j];
      else
         ptable(x - mT_min) = 0;
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
   const int jmax = min(mT_max, rho);
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
      if (i + mT_min <= 0)
         {
         // Fthis[0] = Fprev[0] * Pval_d;
         F[0] = Fprev * Pval_d;
         }
      // determine limits for remaining columns (after first)
      const int jmin = max(i + mT_min, 1);
      const int jmax = min(i + mT_max, rho);
      // keep Fprev[jmin - 1], if necessary
      if (jmin > 1)
         Fprev = F[jmin - 1];
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
         if (j < i + mT_max)
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
   if (i + mT_min <= 0)
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
      if (j < i + mT_max)
         // temp += Fprev[j] * Pval_d;
         temp += Fprev * Pval_d;
      // store result
      // Fthis[j] = temp;
      F[j] = temp;
      }
   // copy results and return
   assertalways(ptable.size() == mT_max - mT_min + 1);
   for (int x = mT_min; x <= mT_max; x++)
      {
      // convert index
      const int j = x + n;
      if (j >= 0 && j <= rho)
         ptable(x - mT_min) = F[j];
      else
         ptable(x - mT_min) = 0;
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
   // set block size to unusable value
   T = 0;
   // initialize metric computer
   computer.init();
   }

/*!
 * \brief Resize drift pdf table
 *
 * The input pdf table can be any size, with any offset; the data is copied
 * into the output pdf table, going from mT_min to mT_max.
 */
template <class G, class real>
libbase::vector<double> qids<G, real>::resize_drift(const array1d_t& in,
      const int offset, const int mT_min, const int mT_max)
   {
   // allocate space an initialize
   array1d_t out(mT_max - mT_min + 1);
   out = 0;
   // copy over common elements
   const int imin = std::max(-offset, mT_min);
   const int imax = std::min(in.size() - 1 - offset, mT_max);
   const int length = imax - imin + 1;
#if DEBUG>=3
   std::cerr << "DEBUG (qids): [resize] offset = " << offset << "." << std::endl;
   std::cerr << "DEBUG (qids): [resize] mT_min = " << mT_min << ", mT_max = " << mT_max << "." << std::endl;
   std::cerr << "DEBUG (qids): [resize] imin = " << imin << ", imax = " << imax << "." << std::endl;
#endif
   out.segment(imin - mT_min, length) = in.extract(imin + offset, length);
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
   int mT_min, mT_max;
   compute_limits(tau, Pr, mT_min, mT_max);
   // store the necessary offset
   offset = libbase::size_type<libbase::vector>(-mT_min);
   // initialize result vector
   eof_pdf.init(mT_max - mT_min + 1);
   // compute the probability at each possible drift
   for (int x = mT_min; x <= mT_max; x++)
      {
      eof_pdf(x - mT_min) = qids_utils::compute_drift_prob_exact(x, tau, Pi, Pd);
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
   int mT_min, mT_max;
   compute_limits(tau, Pr, mT_min, mT_max, sof_pdf, offset);
   // initialize result vector
   eof_pdf.init(mT_max - mT_min + 1);
   // compute the probability at each possible drift
   for (int x = mT_min; x <= mT_max; x++)
      {
      eof_pdf(x - mT_min) = qids_utils::compute_drift_prob_with(
            qids_utils::compute_drift_prob_exact, x, tau, Pi, Pd, sof_pdf,
            offset);
      }
   // resize start-of-frame pdf
   sof_pdf = resize_drift(sof_pdf, offset, mT_min, mT_max);
   // update with the new offset
   offset = libbase::size_type<libbase::vector>(-mT_min);
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
   sout << 4 << std::endl;
   sout << "# Vary Ps?" << std::endl;
   sout << varyPs << std::endl;
   sout << "# Vary Pd?" << std::endl;
   sout << varyPd << std::endl;
   sout << "# Vary Pi?" << std::endl;
   sout << varyPi << std::endl;
   sout << "# Cap on m1_max (0=uncapped) [trellis receiver only]" << std::endl;
   sout << Icap << std::endl;
   sout << "# Cap on n/tau max/min (0=uncapped)" << std::endl;
   sout << Scap << std::endl;
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
 *
 * \version 4 Added support for cap on state space limits
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
   // read cap on state space limits, if present
   if (version >= 4)
      sin >> libbase::eatcomments >> Scap >> libbase::verify;
   else
      Scap = 0;
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
