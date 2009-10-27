#ifndef __dminner_h
#define __dminner_h

#include "config.h"

#include "informed_modulator.h"
#include "fba.h"
#include "channel/bsid.h"

#include "bitfield.h"
#include "randgen.h"
#include "itfunc.h"
#include "serializer.h"
#include <stdlib.h>
#include <math.h>
#include <memory>

namespace libcomm {

/*!
 * \brief   Davey-MacKay Inner Code, original bit-level decoding.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Implements 'Watermark' Codes as described by Davey and MacKay in "Reliable
 * Communication over Channels with Insertions, Deletions, and Substitutions",
 * Trans. IT, Feb 2001.
 *
 * \note In demodulate(), the ptable is internally computed as type 'real',
 * and then copied over after normalization. We norm over the whole
 * block instead of independently for each timestep. This should be
 * equivalent to no-normalization, and is a precursor to a change in the
 * architecture to allow higher-range ptables.
 *
 * \todo Separate this class from friendship with dminner2; common elements
 * should be extracted into a common base
 */

template <class real, bool norm>
class dminner2;

template <class real, bool norm>
class dminner : public informed_modulator<bool> ,
      public parametric,
      private fba<real, bool, norm> {
   friend class dminner2<real, norm> ;
private:
   // Shorthand for class hierarchy
   typedef dminner<real, norm> This;
   typedef fba<real, bool, norm> FBA;
public:
   /*! \name Type definitions */
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<bool> array1b_t;
   typedef libbase::vector<double> array1d_t;
   typedef libbase::vector<real> array1r_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   typedef libbase::vector<array1r_t> array1vr_t;
   enum lut_t {
      lut_straight = 0, lut_user
   };
   // @}
private:
   /*! \name User-defined parameters */
   int n; //!< number of bits in sparse (output) symbol
   int k; //!< number of bits in message (input) symbol
   lut_t lut_type; //!< enum indicating LUT type
   std::string lutname; //!< name to describe codebook
   array1i_t lut; //!< sparsifier LUT
   bool user_threshold; //!< flag indicating that thresholds are supplied by user
   double th_inner; //!< Threshold factor for inner cycle
   double th_outer; //!< Threshold factor for outer cycle
   // @}
   /*! \name Pre-computed parameters */
   double f; //!< average weight per bit of sparse symbol
   // @}
   /*! \name Internally-used objects */
   bsid mychan; //!< bound channel object
   mutable libbase::randgen r; //!< watermark sequence generator
   mutable array1i_t ws; //!< watermark sequence
   // @}
private:
   // Implementations of channel-specific metrics for fba
   real R(const int i, const array1b_t& r);
   // Atomic modem operations (private as these should never be used)
   const bool modulate(const int index) const
      {
      assert("Function should not be used.");
      return false;
      }
   const int demodulate(const bool& signal) const
      {
      assert("Function should not be used.");
      return 0;
      }
   const int demodulate(const bool& signal, const libbase::vector<double>& app) const
      {
      assert("Function should not be used.");
      return 0;
      }
protected:
   // Interface with derived classes
   void advance() const;
   void domodulate(const int N, const array1i_t& encoded, array1b_t& tx);
   void dodemodulate(const channel<bool>& chan, const array1b_t& rx,
         array1vd_t& ptable);
   void dodemodulate(const channel<bool>& chan, const array1b_t& rx,
         const array1vd_t& app, array1vd_t& ptable);

private:
   /*! \name Internal functions */
   void test_invariant() const;
   int fill(int i = 0, libbase::bitfield suffix = libbase::bitfield(""),
         int weight = -1);
   void copypilot(libbase::vector<libbase::bitfield> pilotb);
   void copylut(libbase::vector<libbase::bitfield> lutb);
   void showlut(std::ostream& sout) const;
   void validatelut() const;
   void computemeandensity();
   void checkforchanges(int I, int xmax) const;
   void work_results(const array1b_t& r, array1vr_t& ptable, const int xmax,
         const int dxmax, const int I) const;
   void normalize_results(const array1vr_t& in, array1vd_t& out) const;
   // @}
protected:
   /*! \name Internal functions */
   void init();
   // @}
public:
   /*! \name Constructors / Destructors */
   explicit dminner(const int n = 2, const int k = 1);
   dminner(const int n, const int k, const double th_inner,
         const double th_outer);
   // @}

   /*! \name Watermark-specific setup functions */
   void set_pilot(libbase::vector<bool> pilot);
   void set_pilot(libbase::vector<libbase::bitfield> pilot);
   void set_lut(libbase::vector<libbase::bitfield> lut);
   void set_thresholds(const double th_inner, const double th_outer);
   void set_parameter(const double x)
      {
      set_thresholds(x, x);
      }
   double get_parameter() const
      {
      assert(th_inner == th_outer);
      return th_inner;
      }
   // @}

   /*! \name Watermark-specific informative functions */
   int get_symbolsize() const
      {
      return n;
      }
   int get_symbol(int i) const
      {
      return lut(i);
      }
   double get_th_inner() const
      {
      return th_inner;
      }
   double get_th_outer() const
      {
      return th_outer;
      }
   // @}

   // Setup functions
   void seedfrom(libbase::random& r)
      {
      this->r.seed(r.ival());
      }

   // Informative functions
   int num_symbols() const
      {
      return 1 << k;
      }
   libbase::size_type<libbase::vector> output_block_size() const
      {
      return libbase::size_type<libbase::vector>(input_block_size() * n);
      }
   double energy() const
      {
      return n;
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(dminner)
};

template <class real, bool norm>
inline void dminner<real, norm>::test_invariant() const
   {
   // check code parameters
   assert(n >= 1 && n <= 32);
   assert(k >= 1 && k <= n);
   // check cutoff thresholds
   assert(th_inner >= 0 && th_inner <= 1);
   assert(th_outer >= 0 && th_outer <= 1);
   }

} // end namespace

#endif
