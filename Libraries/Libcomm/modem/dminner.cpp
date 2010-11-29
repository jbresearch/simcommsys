/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "dminner.h"
#include "timer.h"
#include "pacifier.h"
#include "vectorutils.h"
#include <sstream>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show LUTs on manual update
// 3 - Show
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 3
#endif

// internally-used functions

/*!
 * \brief Set up LUT with the lowest weight codewords
 */
template <class real, bool norm>
int dminner<real, norm>::fill(int i, libbase::bitfield suffix, int w)
   {
   assert(lut_type == lut_straight);
   using libbase::bitfield;
   // set up if this is the first (root) call
   if (i == 0 && w == -1)
      {
      lut.init(num_symbols());
      suffix = bitfield("");
      w = n;
      }
   // stop here if we've reached the end
   if (i >= lut.size())
      return i;
   // otherwise, it all depends on the weight we're considering
   bitfield b;
#ifndef NDEBUG
   if (n > 2)
      libbase::trace << "Starting fill with:\t" << suffix << "\t" << w << std::endl;
#endif
   if (w == 0)
      lut(i++) = suffix;
   else
      {
      w--;
      if (suffix.size() == 0)
         i = fill(i, suffix, w);
      for (b = bitfield("1"); b.size() + suffix.size() + w <= n; b = b
            + bitfield("0"))
         i = fill(i, b + suffix, w);
      }
   return i;
   }

/*!
 * \brief Set up pilot sequence for the current frame as given
 */
template <class real, bool norm>
void dminner<real, norm>::copypilot(libbase::vector<libbase::bitfield> pilotb)
   {
   assertalways(pilotb.size() > 0);
   // initialize LUT
   ws.init(pilotb.size());
   // copy elements
   for (int i = 0; i < ws.size(); i++)
      {
      assertalways(pilotb(i).size() == n);
      ws(i) = pilotb(i);
      }
   }

/*!
 * \brief Set up LUT with the given codewords
 */
template <class real, bool norm>
void dminner<real, norm>::copylut(libbase::vector<libbase::bitfield> lutb)
   {
   assertalways(lutb.size() == num_symbols());
   // initialize LUT
   lut.init(num_symbols());
   // copy elements
   for (int i = 0; i < lut.size(); i++)
      {
      assertalways(lutb(i).size() == n);
      lut(i) = lutb(i);
      }
   }

/*!
 * \brief Display LUT on given stream
 */

template <class real, bool norm>
void dminner<real, norm>::showlut(std::ostream& sout) const
   {
   sout << "LUT (k=" << k << ", n=" << n << "):" << std::endl;
   for (int i = 0; i < lut.size(); i++)
      sout << i << "\t" << libbase::bitfield(lut(i), n) << "\t"
            << libbase::weight(lut(i)) << std::endl;
   }

/*!
 * \brief Confirm that LUT is valid
 * Checks that all LUT entries are within range and that there are no
 * duplicate entries.
 */

template <class real, bool norm>
void dminner<real, norm>::validatelut() const
   {
   assertalways(lut.size() == num_symbols());
   for (int i = 0; i < lut.size(); i++)
      {
      // all entries should be within size
      assertalways(lut(i) >= 0 && lut(i) < (1<<n));
      // all entries should be distinct
      for (int j = 0; j < i; j++)
         assertalways(lut(i) != lut(j));
      }
   }

//! Compute and update mean density of sparse alphabet

template <class real, bool norm>
void dminner<real, norm>::computemeandensity()
   {
   array1i_t w = lut;
   w.apply(libbase::weight);
   f = w.sum() / double(n * w.size());
#ifndef NDEBUG
   if (n > 2)
      libbase::trace << "Watermark code density = " << f << std::endl;
#endif
   }

//! Inform user if I or xmax have changed

template <class real, bool norm>
void dminner<real, norm>::checkforchanges(int I, int xmax) const
   {
   static int last_I = 0;
   static int last_xmax = 0;
   if (last_I != I || last_xmax != xmax)
      {
      std::cerr << "DMinner: I = " << I << ", xmax = " << xmax << std::endl;
      last_I = I;
      last_xmax = xmax;
      }
   }

template <class real, bool norm>
void dminner<real, norm>::work_results(const array1b_t& r, array1vr_t& ptable,
      const int xmax, const int dxmax, const int I) const
   {
   libbase::pacifier progress("FBA Results");
   // local flag for path thresholding
   const bool thresholding = (th_outer > 0);
   // determine limits
   const int dmin = std::max(-n, -dxmax);
   const int dmax = std::min(n * I, dxmax);
   // Inherit block size from last modulation step
   const int q = 1 << k;
   const int N = ws.size();
   // Initialise result vector (one sparse symbol per timestep)
   libbase::allocate(ptable, N, q);
   // ptable(i,d) is the a posteriori probability of having transmitted symbol 'd' at time 'i'
   for (int i = 0; i < N; i++)
      {
      std::cerr << progress.update(i, N);
      // determine the strongest path at this point
      real threshold = 0;
      if (thresholding)
         {
         for (int x1 = -xmax; x1 <= xmax; x1++)
            if (FBA::getF(n * i, x1) > threshold)
               threshold = FBA::getF(n * i, x1);
         threshold *= th_outer;
         }
      for (int d = 0; d < q; d++)
         {
         real p = 0;
         // create the considered transmitted sequence
         array1b_t t(n);
         for (int j = 0, tval = ws(i) ^ lut(d); j < n; j++, tval >>= 1)
            t(j) = (tval & 1);
         // event must fit the received sequence:
         // (this is limited to start and end conditions)
         // 1. n*i+x1 >= 0
         // 2. n*(i+1)-1+x2 <= r.size()-1
         // limits on insertions and deletions must be respected:
         // 3. x2-x1 <= n*I
         // 4. x2-x1 >= -n
         // limits on introduced drift in this section:
         // (necessary for forward recursion on extracted segment)
         // 5. x2-x1 <= dxmax
         // 6. x2-x1 >= -dxmax
         const int x1min = std::max(-xmax, -n * i);
         const int x1max = xmax;
         const int x2max_bnd = std::min(xmax, r.size() - n * (i + 1));
         for (int x1 = x1min; x1 <= x1max; x1++)
            {
            const real F = FBA::getF(n * i, x1);
            // ignore paths below a certain threshold
            if (thresholding && F < threshold)
               continue;
            const int x2min = std::max(-xmax, dmin + x1);
            const int x2max = std::min(x2max_bnd, dmax + x1);
            for (int x2 = x2min; x2 <= x2max; x2++)
               {
               // compute the conditional probability
               const real R = mychan.receive(t, r.extract(n * i + x1, x2 - x1
                     + n));
               const real B = FBA::getB(n * (i + 1), x2);
               // include the probability for this particular sequence
               p += F * R * B;
               }
            }
         ptable(i)(d) = p;
         }
      }
   if (N > 0)
      std::cerr << progress.update(N, N);
   }

/*!
 * \brief Normalize probability table
 * 
 * The input probability table is normalized such that the largest value is
 * equal to 1; result is converted to double.
 */
template <class real, bool norm>
void dminner<real, norm>::normalize_results(const array1vr_t& in,
      array1vd_t& out) const
   {
   const int N = in.size();
   assert(N > 0);
   const int q = in(0).size();
   // check for numerical underflow
   real scale = 0;
   for (int i = 0; i < N; i++)
      scale = std::max(scale, in(i).max());
   assert(scale != real(0));
   // allocate result space
   libbase::allocate(out, N, q);
   // normalize and copy results
   for (int i = 0; i < N; i++)
      for (int d = 0; d < q; d++)
         out(i)(d) = in(i)(d) / scale;
   }

// initialization / de-allocation

template <class real, bool norm>
void dminner<real, norm>::init()
   {
   // Fill default LUT if necessary
   if (lut_type == lut_straight)
      fill();
#ifndef NDEBUG
   // Display LUT when debugging
   if (n > 2)
      showlut(libbase::trace);
#endif
   // Validate LUT and compute the mean density
   validatelut();
   computemeandensity();
   // set default thresholds if necessary
   if (!user_threshold)
      {
      th_inner = 1e-15;
      th_outer = 1e-6;
      }
   // Seed the watermark generator and clear the sequence
   r.seed(0);
   ws.init(0);
   // Check that everything makes sense
   test_invariant();
   }

// constructor / destructor

template <class real, bool norm>
dminner<real, norm>::dminner(const int n, const int k) :
   n(n), k(k), lut_type(lut_straight), user_threshold(false)
   {
   init();
   }

template <class real, bool norm>
dminner<real, norm>::dminner(const int n, const int k, const double th_inner,
      const double th_outer) :
   n(n), k(k), lut_type(lut_straight), user_threshold(true),
         th_inner(th_inner), th_outer(th_outer)
   {
   init();
   }

// Watermark-specific setup functions

/*!
 * \copydoc set_pilot()
 * \todo Consider moving this method to the dminner2d class
 */
template <class real, bool norm>
void dminner<real, norm>::set_pilot(libbase::vector<bool> pilot)
   {
   assertalways((pilot.size() % n) == 0);
   // init space for converted vector
   libbase::vector<libbase::bitfield> pilotb(pilot.size() / n);
   // convert pilot sequence
   for (int i = 0; i < pilotb.size(); i++)
      pilotb(i) = libbase::bitfield(pilot.extract(i * n, n));
   // pass through the standard method for setting pilot sequence
   set_pilot(pilotb);
   }

/*!
 * \brief Overrides the internally-generated pilot sequence with given one
 * 
 * The intent of this method is to allow users to apply the dminner decoder
 * in derived algorithms, such as the 2D extension.
 * 
 * \todo merge with copypilot()
 */
template <class real, bool norm>
void dminner<real, norm>::set_pilot(libbase::vector<libbase::bitfield> pilot)
   {
   copypilot(pilot);
   }

/*!
 * \brief Overrides the sparse alphabet with given one
 * 
 * The intent of this method is to allow users to apply the dminner decoder
 * in derived algorithms, such as the 2D extension.
 */
template <class real, bool norm>
void dminner<real, norm>::set_lut(libbase::vector<libbase::bitfield> lut)
   {
   copylut(lut);
   computemeandensity();
#if DEBUG>=2
   showlut(libbase::trace);
#endif
   }

template <class real, bool norm>
void dminner<real, norm>::set_thresholds(const double th_inner,
      const double th_outer)
   {
   user_threshold = true;
   This::th_inner = th_inner;
   This::th_outer = th_outer;
   test_invariant();
   }

// implementations of channel-specific metrics for fba

template <class real, bool norm>
real dminner<real, norm>::R(const int i, const array1b_t& r)
   {
   // 'tx' is a matrix of all possible transmitted symbols
   // we know exactly what was transmitted at this timestep
   const int word = i / n;
   const int bit = i % n;
   bool t = ((ws(word) >> bit) & 1);
   // compute the conditional probability
   return mychan.receive(t, r);
   }

// block advance operation - update watermark sequence

template <class real, bool norm>
void dminner<real, norm>::advance() const
   {
   // Inherit sizes
   const int tau = this->input_block_size();
   // Advance pilot sequence only for non-zero block sizes
   if (tau > 0)
      {
      // Initialize space
      ws.init(tau);
      // creates 'tau' elements of 'n' bits each
      for (int i = 0; i < tau; i++)
         ws(i) = r.ival(1 << n);
      }
   }

// encoding and decoding functions

template <class real, bool norm>
void dminner<real, norm>::domodulate(const int N, const array1i_t& encoded,
      array1b_t& tx)
   {
   // TODO: when N is removed from the interface, rename 'tau' to 'N'
   // Inherit sizes
   const int q = 1 << k;
   const int tau = this->input_block_size();
   // Check validity
   assertalways(tau == encoded.size());
   // Each 'encoded' symbol must be representable by a single sparse vector
   assertalways(N == q);
   // Initialise result vector (one bit per sparse vector)
   tx.init(n * tau);
   assertalways(ws.size() == tau);
   // Encode source stream
   for (int i = 0; i < tau; i++)
      {
      const int s = lut(encoded(i) & (q - 1)); // sparse vector
      const int w = ws(i); // watermark vector
#if DEBUG>=3
      libbase::trace << "DEBUG (dminner::modulate): word " << i << "\t";
      libbase::trace << "s = " << libbase::bitfield(s, n) << "\t";
      libbase::trace << "w = " << libbase::bitfield(w, n) << std::endl;
#endif
      // NOTE: we transmit the low-order bits first
      for (int bit = 0, t = s ^ w; bit < n; bit++, t >>= 1)
         tx(i * n + bit) = (t & 1);
      }
   }

template <class real, bool norm>
void dminner<real, norm>::dodemodulate(const channel<bool>& chan,
      const array1b_t& rx, array1vd_t& ptable)
   {
   // Inherit sizes
   const int N = this->input_block_size();
   const int tau = N * n;
   assert(N > 0);
   // Copy channel for access within R()
   mychan = dynamic_cast<const bsid&> (chan);
   // Update substitution probability to take into account sparse addition
   const double Ps = mychan.get_ps();
   mychan.set_ps(Ps * (1 - f) + (1 - Ps) * f);
   // Set block size for main forward-backward pass
   mychan.set_blocksize(tau);
   // Determine required FBA parameter values
   const int I = mychan.compute_I(tau);
   const int xmax = mychan.compute_xmax(tau);
   const int dxmax = mychan.compute_xmax(n);
   checkforchanges(I, xmax);
   // Initialize & perform forward-backward algorithm
   FBA::init(tau, I, xmax, th_inner);
   FBA::prepare(rx);
   // Reset substitution probability to original value
   mychan.set_ps(Ps);
   // Set block size for results-computation pass to q-ary symbol size
   mychan.set_blocksize(n);
   // Compute and normalize results
   array1vr_t p;
   work_results(rx, p, xmax, dxmax, I);
   normalize_results(p, ptable);
   }

template <class real, bool norm>
void dminner<real, norm>::dodemodulate(const channel<bool>& chan,
      const array1b_t& rx, const array1vd_t& app, array1vd_t& ptable)
   {
   array1vd_t p;
   // Apply standard demodulation
   This::dodemodulate(chan, rx, p);
   // If we have no prior information, copy results over
   if (app.size() == 0)
      {
      ptable = p;
      return;
      }
   // Multiply-in a-priori probabilities
   const int N = p.size();
   assert(N > 0);
   const int q = p(0).size();
   assert(app.size() == N);
   for (int i = 0; i < N; i++)
      {
      assert(app(i).size() == q);
      for (int d = 0; d < q; d++)
         ptable(i)(d) = p(i)(d) * app(i)(d);
      }
   }

// description output

template <class real, bool norm>
std::string dminner<real, norm>::description() const
   {
   std::ostringstream sout;
   sout << "DM Inner Code (" << n << "/" << k << ", ";
   switch (lut_type)
      {
      case lut_straight:
         sout << "sequential codebook";
         break;

      case lut_user:
         sout << lutname << " codebook";
         break;
      }
   if (user_threshold)
      sout << ", thresholds " << th_inner << "/" << th_outer;
   if (norm)
      sout << ", normalized";
   sout << ")";
   return sout.str();
   }

// object serialization - saving

template <class real, bool norm>
std::ostream& dminner<real, norm>::serialize(std::ostream& sout) const
   {
   sout << user_threshold << std::endl;
   if (user_threshold)
      {
      sout << th_inner << std::endl;
      sout << th_outer << std::endl;
      }
   sout << n << std::endl;
   sout << k << std::endl;
   sout << lut_type << std::endl;
   if (lut_type == lut_user)
      {
      sout << lutname << std::endl;
      assert(lut.size() == num_symbols());
      for (int i = 0; i < lut.size(); i++)
         sout << libbase::bitfield(lut(i), n) << std::endl;
      }
   return sout;
   }

// object serialization - loading

template <class real, bool norm>
std::istream& dminner<real, norm>::serialize(std::istream& sin)
   {
   std::streampos start = sin.tellg();
   sin >> libbase::eatcomments >> user_threshold;
   // deal with inexistent flag as 'false'
   if (sin.fail())
      {
      sin.clear();
      sin.seekg(start);
      user_threshold = false;
      }
   // read or set default thresholds
   if (user_threshold)
      {
      sin >> libbase::eatcomments >> th_inner;
      sin >> libbase::eatcomments >> th_outer;
      }
   sin >> libbase::eatcomments >> n;
   sin >> libbase::eatcomments >> k;
   int temp;
   sin >> libbase::eatcomments >> temp;
   lut_type = (lut_t) temp;
   if (lut_type == lut_user)
      {
      sin >> libbase::eatcomments >> lutname;
      // read LUT from stream
      libbase::vector<libbase::bitfield> lutb;
      lutb.init(num_symbols());
      lutb.serialize(sin);
      // use read LUT
      copylut(lutb);
      }
   init();
   return sin;
   }

} // end namespace

// Explicit Realizations

#include "logrealfast.h"

namespace libcomm {

using libbase::logrealfast;

using libbase::serializer;

template class dminner<logrealfast, false> ;
template <>
const serializer dminner<logrealfast, false>::shelper =
      serializer("blockmodem", "dminner<logrealfast>", dminner<logrealfast,
            false>::create);

template class dminner<double, true> ;
template <>
const serializer dminner<double, true>::shelper = serializer("blockmodem",
      "dminner<double>", dminner<double, true>::create);

template class dminner<float, true> ;
template <>
const serializer dminner<float, true>::shelper = serializer("blockmodem",
      "dminner<float>", dminner<float, true>::create);

} // end namespace
