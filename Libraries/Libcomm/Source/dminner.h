#ifndef __dminner_h
#define __dminner_h

#include "config.h"

#include "informed_modulator.h"
#include "fba.h"
#include "bsid.h"

#include "bitfield.h"
#include "randgen.h"
#include "itfunc.h"
#include "serializer.h"
#include <stdlib.h>
#include <math.h>
#include <memory>

namespace libcomm {

/*!
   \brief   Davey's Watermark Code.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   Implements Watermark Codes as described by Davey in "Reliable Communication
   over Channels with Insertions, Deletions, and Substitutions", Trans. IT,
   Feb 2001.

   \todo Make demodulation independent of the previous modulation step.
         Current model assumes a cyclic modulation/demodulation system, as is
         presently being used in the commsys class. This requires:
         - a member variable that keeps the last transmitted block
         - functions to create last transmitted block

   \note In demodulate(), the ptable is internally computed as type 'real',
         and then copied over after normalization. We normalize over the whole
         block instead of independently for each timestep. This should be
         equivalent to no-normalization, and is a precursor to a change in the
         architecture to allow higher-range ptables.
*/

template <class real, bool normalize>
class dminner2;

template <class real, bool normalize>
class dminner : public informed_modulator<bool>, parametric, private fba<real,bool,normalize> {
   friend class dminner2<real,normalize>;
private:
   /*! \name Internally-used types */
   typedef libbase::vector<int>     array1i_t;
   typedef libbase::vector<bool>    array1b_t;
   typedef libbase::matrix<double>  array2d_t;
   typedef libbase::matrix<real>    array2r_t;
   typedef boost::assignable_multi_array<double,1> array1d_t;
   // @}
private:
   /*! \name User-defined parameters */
   int      n;                //!< number of bits in sparse (output) symbol
   int      k;                //!< number of bits in message (input) symbol
   bool     user_lut;         //!< flag indicating that LUT is supplied by user
   std::string lutname;       //!< name to describe codebook
   array1i_t lut;             //!< sparsifier LUT
   bool     user_threshold;   //!< flag indicating that LUT is supplied by user
   double   th_inner;         //!< Threshold factor for inner cycle
   double   th_outer;         //!< Threshold factor for outer cycle
   // @}
   /*! \name Pre-computed parameters */
   double   f;                //!< average weight per bit of sparse symbol
   // @}
   /*! \name Internally-used objects */
   bsid     mychan;           //!< bound channel object
   libbase::randgen r;        //!< watermark sequence generator
   array1i_t ws;              //!< watermark sequence
   // @}
private:
   /*! \name Internal functions */
   void test_invariant() const;
   int fill(int i=0, libbase::bitfield suffix="", int weight=-1);
   void createsequence(const int tau);                      
   void checkforchanges(int I, int xmax) const;   
   void work_results(const array1b_t& r, array2r_t& ptable, const int xmax, const int dxmax, const int I) const;
   void normalize_results(const array2r_t& in, array2d_t& out) const;
   // @}
   // Implementations of channel-specific metrics for fba
   real R(const int i, const array1b_t& r);
   // Atomic modem operations (private as these should never be used)
   const bool modulate(const int index) const
      { assert("Function should not be used."); return false; };
   const int demodulate(const bool& signal) const
      { assert("Function should not be used."); return 0; };
   const int demodulate(const bool& signal, const libbase::vector<double>& app) const
      { assert("Function should not be used."); return 0; };
protected:
   /*! \name Internal functions */
   void init();
   // @}
public:
   /*! \name Constructors / Destructors */
   dminner(const int n=2, const int k=1);
   dminner(const int n, const int k, const double th_inner, const double th_outer);
   // @}

   /*! \name Watermark-specific setup functions */
   void set_thresholds(const double th_inner, const double th_outer);
   void set_parameter(const double x) { set_thresholds(x,x); };
   double get_parameter() const { assert(th_inner==th_outer); return th_inner; };
   // @}

   /*! \name Watermark-specific informative functions */
   int get_n() const { return n; };
   int get_k() const { return k; };
   int get_lut(int i) const { return lut(i); };
   double get_th_inner() const { return th_inner; };
   double get_th_outer() const { return th_outer; };
   // @}

   // Vector modem operations
   void modulate(const int N, const array1i_t& encoded, array1b_t& tx);
   void demodulate(const channel<bool>& chan, const array1b_t& rx, array2d_t& ptable);
   void demodulate(const channel<bool>& chan, const array1b_t& rx, const array2d_t& app, array2d_t& ptable);

   // Setup functions
   void seedfrom(libbase::random& r) { this->r.seed(r.ival()); };

   // Informative functions
   int num_symbols() const { return 1<<k; };
   double energy() const { return n; };

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(dminner)
};

template <class real, bool normalize>
inline void dminner<real,normalize>::test_invariant() const
   {
   // check code parameters
   assert(k >= 1);
   assert(n > k);
   // check cutoff thresholds
   assert(th_inner >= 0 && th_inner <= 1);
   assert(th_outer >= 0 && th_outer <= 1);
   }

}; // end namespace

#endif
