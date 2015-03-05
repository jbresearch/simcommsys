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

#ifndef __qids_h
#define __qids_h

#include "config.h"
#include "matrix.h"
#include "qids-utils.h"
#include "channel_stream.h"
#include "field_utils.h"
#include "serializer.h"
#include "cuda-all.h"

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show tx and rx vectors when computing RecvPr
// NOTE: since this is a header, it may be included in other classes as well;
//       to avoid problems, the debug level is reset at the end of this file.
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*!
 * \brief   q-ary insertion/deletion/substitution channel.
 * \author  Johann Briffa
 *
 * Implements a q-ary extension of the channel with unbounded random insertion,
 * deletion and substitution errors as described in:
 * Davey, M.C. and Mackay, D.J.C., "Reliable communication over channels with
 * insertions, deletions, and substitutions," IEEE Transactions on Information
 * Theory, vol.47, no.2, pp.687-698, Feb 2001
 * URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=910582&isnumber=19638
 *
 * \tparam G Channel symbol type
 * \tparam real Floating-point type for internal computation
 */

template <class G, class real>
class qids : public channel_stream<G, real> {
public:
   /*! \name Type definitions */
   typedef libbase::matrix<real> array2r_t;
   typedef libbase::vector<real> array1r_t;
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<bool> array1b_t;
   typedef libbase::vector<G> array1g_t;
   typedef libbase::vector<double> array1d_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   enum receiver_t {
      receiver_trellis = 0, //!< trellis-based receiver
      receiver_lattice, //!< lattice-based receiver without constraints
      receiver_lattice_corridor, //!< lattice-based receiver with drift constraint
      receiver_undefined
   };
   // @}
private:
   /*! \name User-defined parameters */
   bool varyPs; //!< Flag to indicate that \f$ P_s \f$ should change with parameter
   bool varyPd; //!< Flag to indicate that \f$ P_d \f$ should change with parameter
   bool varyPi; //!< Flag to indicate that \f$ P_i \f$ should change with parameter
   int Icap; //!< Maximum usable value of m1_max (0 indicates no cap is placed)
   int Scap; //!< Maximum usable value of n/tau max/min (0 indicates no cap is placed)
   double fixedPs; //!< Value to use when \f$ P_s \f$ does not change with parameter
   double fixedPd; //!< Value to use when \f$ P_d \f$ does not change with parameter
   double fixedPi; //!< Value to use when \f$ P_i \f$ does not change with parameter
   // @}
public:
   /*! \name Metric computation */
   class metric_computer : public channel_insdel<G, real>::metric_computer {
   public:
      /*! \name User-defined parameters */
      receiver_t receiver_type; //!< enum indicating receiver implementation to use
      // @}
      /*! \name Channel-state and pre-computed parameters */
#ifdef USE_CUDA
      cuda::matrix_auto<real> Rtable; //!< Receiver coefficient set for mu >= 0
#else
      array2r_t Rtable; //!< Receiver coefficient set for mu >= 0
#endif
      real Rval; //!< Receiver coefficient value for mu = -1
      real Pval_d; //!< Lattice coefficient value for deletion event
      real Pval_i; //!< Lattice coefficient value for insertion event
      real Pval_tc; //!< Lattice coefficient value for correct transmission event
      real Pval_te; //!< Lattice coefficient value for error transmission event
      int T; //!< block size in channel symbols
      int mT_min; //!< Assumed largest negative drift over a whole \c T channel-symbol block is \f$ m_T^{-} \f$
      int mT_max; //!< Assumed largest positive drift over a whole \c T channel-symbol block is \f$ m_T^{+} \f$
      int m1_min; //!< Assumed largest negative drift over a single channel symbol is \f$ m_1^{-} \f$
      int m1_max; //!< Assumed largest positive drift over a single channel symbol is \f$ m_1^{+} \f$
      // @}
      /*! \name Hardwired parameters */
      static const int arraysize = 2 * 63 + 1; //!< Size of stack-allocated arrays
      // @}
   public:
      /*! \name FBA decoder parameter computation */
      // receiver metric pre-computation
      static real compute_Rtable_entry(bool err, int mu, double Ps, double Pd,
            double Pi);
      static void compute_Rtable(array2r_t& Rtable, int m1_max, double Ps, double Pd,
            double Pi);
      // @}
   public:
      /*! \name Constructors / Destructors */
      metric_computer() :
      receiver_type(receiver_trellis)
         {
         }
      // @}
      /*! \name Internal functions */
      void precompute(double Ps, double Pd, double Pi, int T, int mT_min,
            int mT_max, int m1_min, int m1_max);
      void init()
         {
#ifdef USE_CUDA
         // Initialize CUDA
         cuda::cudaInitialize(std::cerr);
#endif
         }
      // @}
#ifdef USE_CUDA
      /*! \name Device methods */
#ifdef __CUDACC__
      //! Receiver interface
      __device__
      real receive(const cuda::vector_reference<G>& tx, const cuda::vector_reference<G>& rx) const
         {
         // Compute sizes
         const int n = tx.size();
         const int mu = rx.size() - n;
         // Allocate space for results and call main receiver
         real ptable_data[arraysize];
         cuda_assertalways(arraysize >= mT_max - mT_min + 1);
         cuda::vector_reference<real> ptable(ptable_data, mT_max - mT_min + 1);
         receive(tx, rx, ptable);
         // return result
         return ptable(mu - mT_min);
         }
      //! Batch receiver interface
      __device__
      void receive(const cuda::vector_reference<G>& tx, const cuda::vector_reference<G>& rx,
            cuda::vector_reference<real>& ptable) const
         {
         switch(receiver_type)
            {
            case receiver_trellis:
               receive_trellis(tx, rx, ptable);
               break;
            case receiver_lattice:
               receive_lattice(tx, rx, ptable);
               break;
            case receiver_lattice_corridor:
               receive_lattice_corridor(tx, rx, ptable);
               break;
            default:
               cuda_failwith("Unknown receiver mode");
               break;
            }
         }
      //! Batch receiver interface - trellis computation
      __device__
      void receive_trellis(const cuda::vector_reference<G>& tx, const cuda::vector_reference<G>& rx,
            cuda::vector_reference<real>& ptable) const
         {
         using cuda::min;
         using cuda::max;
         using cuda::swap;
         // Compute sizes
         const int n = tx.size();
         const int rho = rx.size();
         //cuda_assert(n <= T);
         assert(rho - n <= mT_max);
         assert(rho - n >= mT_min);
         // Set up two slices of forward matrix, and associated pointers
         // Arrays are allocated on the stack as a fixed size; this avoids dynamic
         // allocation (which would otherwise be necessary as the size is non-const)
         cuda_assertalways(mT_max - mT_min + 1 <= arraysize);
         real F0[arraysize];
         real F1[arraysize];
         __restrict__ real* Fthis = F1 - mT_min; // offset: first element is index 'mT_min'
         __restrict__ real* Fprev = F0 - mT_min; // offset: first element is index 'mT_min'
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
         cuda_assertalways(ptable.size() == mT_max - mT_min + 1);
         for (int x = mT_min; x <= mT_max; x++)
            ptable(x - mT_min) = Fthis[x];
         }
      //! Batch receiver interface - lattice computation
      __device__
      void receive_lattice(const cuda::vector_reference<G>& tx, const cuda::vector_reference<G>& rx,
            cuda::vector_reference<real>& ptable) const
         {
         using cuda::swap;
         // Compute sizes
         const int n = tx.size();
         const int rho = rx.size();
         // Set up two slices of lattice, and associated pointers
         // Arrays are allocated on the stack as a fixed size; this avoids dynamic
         // allocation (which would otherwise be necessary as the size is non-const)
         cuda_assertalways(rho + 1 <= arraysize);
         real F0[arraysize];
         real F1[arraysize];
         __restrict__ real* Fthis = F1;
         __restrict__ real* Fprev = F0;
         // initialize for i=0 (first row of lattice)
         Fthis[0] = 1;
         for (int j = 1; j <= rho; j++)
            {
            Fthis[j] = Fthis[j - 1] * Pval_i;
            }
         // compute remaining rows, except last
         for (int i = 1; i < n; i++)
            {
            // swap 'this' and 'prior' rows
            swap(Fthis, Fprev);
            // handle first column as a special case
            Fthis[0] = Fprev[0] * Pval_d;
            // remaining columns
            for (int j = 1; j <= rho; j++)
               {
               const double pi = Fthis[j - 1] * Pval_i;
               const double pd = Fprev[j] * Pval_d;
               const bool cmp = tx(i - 1) == rx(j - 1);
               const double ps = Fprev[j - 1] * (cmp ? Pval_tc : Pval_te);
               Fthis[j] = pi + ps + pd;
               }
            }
         // compute last row as a special case (no insertions)
         // swap 'this' and 'prior' rows
         swap(Fthis, Fprev);
         // handle first column as a special case
         Fthis[0] = Fprev[0] * Pval_d;
         // remaining columns
         for (int j = 1; j <= rho; j++)
            {
            const double pd = Fprev[j] * Pval_d;
            const bool cmp = tx(n - 1) == rx(j - 1);
            const double ps = Fprev[j - 1] * (cmp ? Pval_tc : Pval_te);
            Fthis[j] = ps + pd;
            }
         // copy results and return
         cuda_assertalways(ptable.size() == mT_max - mT_min + 1);
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
      //! Batch receiver interface - lattice corridor computation
      __device__
      void receive_lattice_corridor(const cuda::vector_reference<G>& tx, const cuda::vector_reference<G>& rx,
            cuda::vector_reference<real>& ptable) const
         {
         using cuda::min;
         using cuda::max;
         using cuda::swap;
         // Compute sizes
         const int n = tx.size();
         const int rho = rx.size();
         // Set up single slice of lattice on the stack as a fixed size;
         // this avoids dynamic allocation (which would otherwise be necessary
         // as the size is non-const)
         /*
         cuda_assertalways(rho + 1 <= arraysize);
         real F[arraysize];
         */
         // set up variable to keep track of Fprev[j-1]
         real Fprev;
         // get access to slice of lattice in shared memory
         cuda::SharedMemory<real> smem;
         const int pitch = n + mT_max + 1;
         cuda_assertalways(rho + 1 <= pitch);
         __restrict__ real* F = smem.getPointer() + (threadIdx.x + threadIdx.y * blockDim.x) * pitch;
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
         cuda_assertalways(ptable.size() == mT_max - mT_min + 1);
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
      // @}
#endif
      /*! \name Host methods */
      //! Determine the amount of shared memory required per thread
      size_t receiver_sharedmem() const
         {
         switch(receiver_type)
            {
            case receiver_trellis:
               return 0;
            case receiver_lattice:
               return 0;
            case receiver_lattice_corridor:
               return (T + mT_max + 1) * sizeof(real);
            default:
               failwith("Unknown receiver mode");
               return 0;
            }
         }
      //! Receiver interface
      real receive(const G& tx, const array1g_t& rx) const
         {
         // Compute sizes
         const int mu = rx.size() - 1;
         // If this was not a deletion, return result from table
         if (mu >= 0)
            {
#ifdef USE_CUDA
            // create local table and copy from device
            array2r_t Rtable_temp;
            Rtable_temp = Rtable;
            return Rtable_temp(tx != rx(mu), mu);
#else
            return Rtable(tx != rx(mu), mu);
#endif
            }
         // If this was a deletion, it's a fixed value
         return Rval;
         }
      //! Receiver interface
      real receive(const array1g_t& tx, const array1g_t& rx) const
         {
         // Compute sizes
         const int n = tx.size();
         const int mu = rx.size() - n;
         // Allocate space for results and call main receiver
         static array1r_t ptable;
         ptable.init(mT_max - mT_min + 1);
         receive(tx, rx, ptable);
         // return result
         return ptable(mu - mT_min);
         }
      //! Batch receiver interface - indefinite state space
      void receive(const array1g_t& tx, const array1g_t& rx, array1r_t& ptable) const
         {
         switch(receiver_type)
            {
            case receiver_trellis:
               receive_trellis(tx, rx, ptable);
               break;
            case receiver_lattice:
               receive_lattice(tx, rx, ptable);
               break;
            case receiver_lattice_corridor:
               receive_lattice_corridor(tx, rx, ptable);
               break;
            default:
               failwith("Unknown receiver mode");
               break;
            }
         }
      //! Batch receiver interface - trellis computation
      void receive_trellis(const array1g_t& tx, const array1g_t& rx, array1r_t& ptable) const;
      //! Batch receiver interface - lattice computation
      void receive_lattice(const array1g_t& tx, const array1g_t& rx,
            array1r_t& ptable) const;
      //! Batch receiver interface - lattice computation, restricted to corridor
      void receive_lattice_corridor(const array1g_t& tx, const array1g_t& rx,
            array1r_t& ptable) const;
      //! Batch receiver interface - fixed state space
      void receive(const array1g_t& tx, const array1g_t& rx, const int S0,
            const int delta0, const bool first, const bool last,
            array1r_t& ptable0, array1r_t& ptable1) const
         {
         failwith("Method not supported.");
         }
      // @}
      DECLARE_CLONABLE(metric_computer)
   };
   // @}
private:
   /*! \name Internal representation */
   metric_computer computer;
   double Ps; //!< Symbol substitution probability \f$ P_s \f$
   double Pd; //!< Symbol deletion probability \f$ P_d \f$
   double Pi; //!< Symbol insertion probability \f$ P_i \f$
   double Pr; //!< Probability of channel event outside chosen limits
   int T; //!< Block size in channel symbols over which we want to synchronize
   array1i_t state_ins; //!< State vector with number of insertions before transmission of bit 'i'
   array1b_t state_tx; //!< State vector with flag indicating transmission of bit 'i'
   // @}
private:
   /*! \name Internal functions */
   void init();
   void precompute()
      {
      if (T > 0)
         {
         int mT_min, mT_max;
         compute_limits(T, Pr, mT_min, mT_max);
         int m1_min, m1_max;
         compute_limits(1, qids_utils::divide_error_probability(Pr, T), m1_min,
               m1_max);
         computer.precompute(Ps, Pd, Pi, T, mT_min, mT_max, m1_min, m1_max);
         }
      }
   static array1d_t resize_drift(const array1d_t& in, const int offset,
         const int mT_min, const int mT_max);
   // @}
protected:
   // Channel function overrides
   G corrupt(const G& s);
   double pdf(const G& tx, const G& rx) const
      {
      return (tx == rx) ? 1 - Ps : Ps / field_utils<G>::elements();
      }
public:
   /*! \name Constructors / Destructors */
   /*!
    * \brief Principal constructor
    *
    * \sa init()
    */
   qids(const bool varyPs = true, const bool varyPd = true, const bool varyPi =
         true) :
         varyPs(varyPs), varyPd(varyPd), varyPi(varyPi), Icap(0), Scap(0), fixedPs(
               0), fixedPd(0), fixedPi(0)
      {
      // channel update flags
      assert(varyPs || varyPd || varyPi);
      // other initialization
      init();
      }
   // @}

   /*! \name FBA decoder parameter computation */
   /*!
    * \copydoc qids_utils::compute_I()
    *
    * \note Provided for use by clients; depends on object parameters
    */
   int compute_I(int tau, double Pr) const
      {
      int I = qids_utils::compute_I(tau, Pi, Pr);
      if (Icap > 0)
         I = std::min(I, Icap);
      return I;
      }
   /*!
    * \copydoc qids_utils::compute_xmax()
    *
    * \note Provided for use by clients; depends on object parameters
    */
   int compute_xmax(int tau, double Pr,
         const libbase::vector<double>& sof_pdf = libbase::vector<double>(),
         const int offset = 0) const
      {
      int xmax = qids_utils::compute_xmax(tau, Pi, Pd, Pr, sof_pdf, offset);
      // cap minimum value
      const int I = compute_I(tau, Pr);
      xmax = std::max(xmax, I);
      if (Scap > 0)
         xmax = std::min(xmax, Scap);
      return xmax;
      }
   // @}

   /*! \name Channel parameter handling */
   void set_parameter(const double p);
   double get_parameter() const;
   // @}

   /*! \name Channel parameter setters */
   //! Set the symbol-substitution probability
   void set_ps(const double Ps)
      {
      assert(Ps >= 0 && Ps <= 0.5);
      this->Ps = Ps;
      }
   //! Set the symbol-deletion probability
   void set_pd(const double Pd)
      {
      assert(Pd >= 0 && Pd <= 1);
      assert(Pi + Pd >= 0 && Pi + Pd <= 1);
      this->Pd = Pd;
      precompute();
      }
   //! Set the symbol-insertion probability
   void set_pi(const double Pi)
      {
      assert(Pi >= 0 && Pi <= 1);
      assert(Pi + Pd >= 0 && Pi + Pd <= 1);
      this->Pi = Pi;
      precompute();
      }
   // @}

   /*! \name Channel parameter getters */
   //! Get the current symbol-substitution probability
   double get_ps() const
      {
      return Ps;
      }
   //! Get the current symbol-deletion probability
   double get_pd() const
      {
      return Pd;
      }
   //! Get the current symbol-insertion probability
   double get_pi() const
      {
      return Pi;
      }
   // @}

   /*! \name Stream-oriented channel characteristics */
   void get_drift_pdf(int tau, double Pr, libbase::vector<double>& eof_pdf,
         libbase::size_type<libbase::vector>& offset) const;
   void get_drift_pdf(int tau, double Pr, libbase::vector<double>& sof_pdf,
         libbase::vector<double>& eof_pdf,
         libbase::size_type<libbase::vector>& offset) const;
   // @}

   // Insertion-deletion channel functions
   int get_drift(int t) const
      {
#ifndef NDEBUG
      // shorthand for length of last transmitted frame
      const int tau = state_ins.size();
      assert(state_tx.size() == tau);
      // sanity check
      assert(t >= 0);
      assert(t <= tau);
#endif
      // accumulate drift up to time 't'
      int drift = 0;
      for (int i = 0; i < t; i++)
         {
         drift += state_ins(i);
         if (!state_tx(i))
            drift--;
         }
      return drift;
      }
   //! Set the probability of channel event outside chosen limits
   void set_pr(const double Pr)
      {
      assert(Pr > 0 && Pr < 1);
      this->Pr = Pr;
      precompute();
      }
   //! Set the block size
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
    * \copydoc qids_utils::compute_limits()
    *
    * \note Provided for use by clients; depends on object parameters
    */
   void compute_limits(int tau, double Pr, int& lower, int& upper,
         const libbase::vector<double>& sof_pdf = libbase::vector<double>(),
         const int offset = 0) const
      {
      qids_utils::compute_limits(tau, Pi, Pd, Pr, lower, upper, sof_pdf, offset);
      if (Scap > 0)
         {
         upper = std::min(upper, Scap);
         lower = std::max(lower, -Scap);
         }
      }
   //! Determine whether the channel model has a fixed state space
   bool is_statespace_fixed() const
      {
      return false;
      }

   // Channel functions
   void transmit(const array1g_t& tx, array1g_t& rx);
   //! \note Used by qids::receive(tx, rx, ptable)
   double receive(const G& tx, const array1g_t& rx) const
      {
#if DEBUG>=2
      libbase::trace << "DEBUG (qids): Computing RecvPr for" << std::endl;
      libbase::trace << "tx = " << tx;
      libbase::trace << "rx = " << rx;
#endif
      const real result = computer.receive(tx, rx);
#if DEBUG>=2
      libbase::trace << "RecvPr = " << result << std::endl;
#endif
      return result;
      }
   /*! \note Used by: direct_blockembedder, ssis, direct_blockmodem,
    * lut_modulator
    */
   void receive(const array1g_t& tx, const array1g_t& rx,
         array1vd_t& ptable) const
      {
      // Compute sizes
      const int M = tx.size();
      // Initialize results vector
      ptable.init(1);
      ptable(0).init(M);
      // Compute results for each possible signal
      for (int x = 0; x < M; x++)
         ptable(0)(x) = qids<G, real>::receive(tx(x), rx);
      }
   //! \note Used by dminner
   double receive(const array1g_t& tx, const array1g_t& rx) const
      {
#if DEBUG>=2
      libbase::trace << "DEBUG (qids): Computing RecvPr for" << std::endl;
      libbase::trace << "tx = " << tx;
      libbase::trace << "rx = " << rx;
#endif
      const real result = computer.receive(tx, rx);
#if DEBUG>=2
      libbase::trace << "RecvPr = " << result << std::endl;
#endif
      return result;
      }

   // Interface for CUDA
   const typename channel_insdel<G, real>::metric_computer& get_computer() const
      {
      return computer;
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(qids)
};

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG
#endif

} // end namespace

#endif
