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

template <class real> class watermarkcode : public modulator<bool>, private fba<real,bool> {
private:
   /*! \name User-defined parameters */
   int      n;                //!< number of bits in sparse (output) symbol
   int      k;                //!< number of bits in message (input) symbol
   bool     userspecified;    //!< flag indicating that LUT is supplied by user
   std::string lutname;       //!< name to describe codebook
   // @}
   /*! \name Pre-computed parameters */
   double   f;    //!< average weight per bit of sparse symbol
   // @}
   /*! \name Internally-used objects */
   bsid mychan;               //!< bound channel object
   libbase::randgen r;        //!< watermark sequence generator
   libbase::vector<int> ws;   //!< watermark sequence
   libbase::vector<int> lut;  //!< sparsifier LUT
   libbase::vector<double> Ptable;  //!< pre-computed values for P function
   // @}
private:
   /*! \name Internal functions */
   int fill(int i=0, libbase::bitfield suffix="", int weight=-1);
   void createsequence(const int tau);                      
   void checkforchanges(int I, int xmax);                      
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
   void free() {};
   // @}
   /*! \name Constructors / Destructors */
   //! Default constructor
   watermarkcode();
   // @}
public:
   /*! \name Constructors / Destructors */
   watermarkcode(const int n, const int k, const bool varyPs=true, const bool varyPd=true, const bool varyPi=true);
   ~watermarkcode() { free(); };
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
   DECLARE_SERIALIZER(watermarkcode)
};

}; // end namespace

#endif
