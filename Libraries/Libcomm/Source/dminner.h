#ifndef __watermarkcode_h
#define __watermarkcode_h

#include "config.h"

#include "modulator.h"
#include "mpsk.h"
#include "fba.h"
#include "bsid.h"

#include "bitfield.h"
#include "randgen.h"
#include "itfunc.h"
#include "serializer.h"
#include <stdlib.h>
#include <math.h>

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
class dminner : public modulator<bool>, private fba<real,bool,normalize> {
   friend class dminner2<real,normalize>;
private:
   /*! \name Internally-used types */
   typedef boost::multi_array<double,1> array1d_t;
   // @}
   /*! \name User-defined parameters */
   int      n;                //!< number of bits in sparse (output) symbol
   int      k;                //!< number of bits in message (input) symbol
   bool     user_lut;         //!< flag indicating that LUT is supplied by user
   std::string lutname;       //!< name to describe codebook
   libbase::vector<int> lut;  //!< sparsifier LUT
   bool     user_threshold;   //!< flag indicating that LUT is supplied by user
   double   th_inner;         //!< Threshold factor for inner cycle
   double   th_outer;         //!< Threshold factor for outer cycle
   // @}
   /*! \name Pre-computed parameters */
   double   f;    //!< average weight per bit of sparse symbol
   // @}
   /*! \name Internally-used objects */
   bsid *mychan;              //!< bound channel object
   libbase::randgen r;        //!< watermark sequence generator
   libbase::vector<int> ws;   //!< watermark sequence
   mutable array1d_t Ptable;  //!< Forward recursion 'P' function lookup
   // @}
private:
   /*! \name Internal functions */
   int fill(int i=0, libbase::bitfield suffix="", int weight=-1);
   void createsequence(const int tau);                      
   void checkforchanges(int I, int xmax) const;                      
   // @}
   // Implementations of channel-specific metrics for fba
   real P(const int a, const int b);
   real Q(const int a, const int b, const int i, const libbase::vector<bool>& s);
   // Atomic modem operations (private as these should never be used)
   const bool modulate(const int index) const { assert("Function should not be used."); return false; };
   const int demodulate(const bool& signal) const { assert("Function should not be used."); return 0; };
protected:
   /*! \name Internal functions */
   void init();
   void free();
   // @}
public:
   /*! \name Constructors / Destructors */
   dminner(const int n=2, const int k=1);
   dminner(const int n, const int k, const double th_inner, const double th_outer);
   ~dminner() { free(); };
   // @}

   /*! \name Watermark-specific informative functions */
   int get_n() const { return n; };
   int get_k() const { return k; };
   int get_lut(int i) const { return lut(i); };
   // @}

   // Vector modem operations
   void modulate(const int N, const libbase::vector<int>& encoded, libbase::vector<bool>& tx);
   void demodulate(const channel<bool>& chan, const libbase::vector<bool>& rx, libbase::matrix<double>& ptable);

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

}; // end namespace

#endif
