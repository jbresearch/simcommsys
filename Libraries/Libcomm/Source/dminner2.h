#ifndef __dminner2_h
#define __dminner2_h

#include "config.h"

#include "modulator.h"
#include "mpsk.h"
#include "fba2.h"
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

   Implements a novel (and more accurate) decoding algorithm for the inner
   codes described by Davey in "Reliable Communication over Channels with
   Insertions, Deletions, and Substitutions", Trans. IT, Feb 2001.
*/

template <class real> class dminner2 : public modulator<bool>, private fba2<real,bool> {
private:
   /*! \name User-defined parameters */
   int      n;                //!< number of bits in sparse (output) symbol
   int      k;                //!< number of bits in message (input) symbol
   bool     userspecified;    //!< flag indicating that LUT is supplied by user
   std::string lutname;       //!< name to describe codebook
   libbase::vector<int> lut;  //!< sparsifier LUT
   // @}
   /*! \name Pre-computed parameters */
   // @}
   /*! \name Internally-used objects */
   bsid *mychan;              //!< bound channel object
   libbase::randgen r;        //!< watermark sequence generator
   libbase::vector<int> ws;   //!< watermark sequence
   // @}
private:
   /*! \name Internal functions */
   int fill(int i=0, libbase::bitfield suffix="", int weight=-1);
   void createsequence(const int tau);                      
   void checkforchanges(int I, int xmax);                      
   // @}
   // Implementations of channel-specific metrics for fba2
   real Q(int d, int i, const libbase::vector<bool>& r) const;
   // Atomic modem operations (private as these should never be used)
   const bool modulate(const int index) const { assert("Function should not be used."); return false; };
   const int demodulate(const bool& signal) const { assert("Function should not be used."); return 0; };
protected:
   /*! \name Internal functions */
   void init();
   void free();
   // @}
   /*! \name Constructors / Destructors */
   //! Default constructor
   dminner2() { mychan = NULL; };
   // @}
public:
   /*! \name Constructors / Destructors */
   dminner2(const int n, const int k);
   ~dminner2() { free(); };
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
   DECLARE_SERIALIZER(dminner2)
};

}; // end namespace

#endif
