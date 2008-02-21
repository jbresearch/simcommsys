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

   \version 1.00 (1-12 Oct 2007)
   initial version; implements Watermark Codes as described by Davey in "Reliable
   Communication over Channels with Insertions, Deletions, and Substitutions",
   Trans. IT, Feb 2001.

   \version 1.10 (25-30 Oct 2007)
   - made class a 'modulator' rather than a 'modulator' as this better reflects its
     position within the communication model's stack
   - removed 'N' from a code parameter - this is simply the size of the block
     currently being demodulated
   - removed 'const' restriction on modulate and demodulate vector functions, as in
     modulator 1.50
   - added private inheritance from bsid channel (to access the transmit and receive
     functions there)
   - updated clone() to return type 'watermarkcode' instead of 'modulator'; this
     avoids a derivation ambiguity problem introduced with the inheritance from bsid,
     which is a channel, not a modulator. [cf. Stroustrup 15.6.2]
   - added assertions during initialization
   - started implementations of P(), Q() and demodulate()
   - added support for keeping track of the last transmitted block; this assumes a cyclic
     modulation/demodulation system, as is presently being used in the commsys class.
     - added member variable that keeps the last transmitted block
     - added functions to create last transmitted block
   - updated to conform with fba 1.10.
   - changed derivation to fba<real> from fba<logrealfast>.
   \todo Make demodulation independent of the previous modulation step.

   \version 1.20 (31 Oct - 1 Nov 2007)
   - updated definition of Q() to conform with fba 1.20
   - implemented Q()
   - implemented demodulate()
   - updated constructor to initialize bsid
   - updated serialization to include all parameters

   \version 1.21 (2 Nov 2007)
   - removed Ps, Pd and Pi from serialization and from construction; also removed
     the variables, as the values should be obtained through the bsid object.
   - now setting defaults for Ps,Pd,Pi to zero in all constructors, through init()
   - added boolean construction parameters varyPs, varyPd, varyPi, as required by
     the bsid class.
   - removed Pf, Pt, alphaI

   \version 1.22 (5 Nov 2007)
   - updated according to the reduced memory usage of F and B matrices, as in
     fba 1.21.
   - updated serialization routines to also serialize the bsid variables (was causing
     problems with I and xmax not being initialized.
   - fixed error in energy(), which was incorrectly returning 1<<n instead of n.

   \version 1.23 (7 Nov 2007)
   - changed bsid from a class derivation to an included object.
   - added debug-mode progress reporting
   - fixed error in demodulate, where the drift introduced by the considered
     sparse symbol was out of bounds.

   \version 1.24 (12-13 Nov 2007)
   - fixed error in modulate(), where it was assumed that each encoded symbol fits
     exactly in a sparse symbol. In fact, each encoded symbol needs to be made up
     of an integral number of sparse symbols.

   \version 1.25 (13 Nov 2007)
   - optimization of demodulate()

   \version 1.26 (26-28 Nov 2007)
   - fixed a bug in modulation, where the incorrect index was applied to the tx vector
   - added debugging information printing during modulation, when working with small blocks
   - added debugging information printing during demodulation, when working with small blocks
   - optimized demodulate() by removing the copying operation on the received sequence
   - fixed a bug in demodulation, where the received vector being considered was
     incorrectly assumed to consist of one bit rather than 'n'.
   - optimized demodulate() by pre-computing loop limits
   - fixed a bug in loop limits, since drift limits were incorrect
   - modified demodulate() so that ptable is internally computed as type 'real', and then
     copied over after normalization.
   - fixed a serious error in demodulate(), where the data element being considered was
     not sparsified before adding to the watermark sequence.
   - added check for numerical underflow in demodulate (debug build).
   - updated to conform with fba 1.30, changing the return type of P() and Q()
     to 'real'.
   - in demodulate(), moved the creation of tx vector two loops outwards, and cleaned it up
   - removed I, xmax from this class, since they are held (and should be only) in bsid channel
   *** first version that actually decodes in a usable way ***

   \version 1.30 (29 Nov 2007)
   - changed normalization method, so that we normalize over the whole block instead of
     independently for each timestep. This should be equivalent to no-normalization, and is
     a precursor to a change in the architecture to allow higher-range ptables.

   \version 1.40 (6 Dec 2007)
   - removed I and xmax from user-defined parameters, as in bsid 1.40

   \version 1.41 (24 Jan 2008)
   - Changed reference from channel to channel<sigspace>

   \version 2.00 (28 Jan 2008)
   - Changed base from mpsk (and therefore modulator<sigspace>) to modulator<bool>

   \version 2.01 (1 Feb 2008)
   - Fixed initial bsid channel creation to use default varyPx values.

   \version 2.02 (21 Feb 2008)
   - Added default values to fill().
   - Moved call to fill() outside init().
*/

template <class real> class watermarkcode : public modulator<bool>, private fba<real,bool> {
   /*! \name Serialization */
   static const libbase::serializer shelper;
   static void* create() { return new watermarkcode<real>; };
   // @}
private:
   /*! \name User-defined parameters */
   int      n;    //!< number of bits in sparse (output) symbol
   int      k;    //!< number of bits in message (input) symbol
   int      s;    //!< watermark generator seed
   // @}
   /*! \name Pre-computed parameters */
   double   f;    //!< average weight per bit of sparse symbol
   // @}
   /*! \name Internally-used objects */
   bsid mychan;               //!< bound channel object
   libbase::randgen r;        //!< watermark sequence generator
   libbase::vector<int> ws;   //!< watermark sequence
   libbase::vector<int> lut;  //!< sparsifier LUT
   // @}
private:
   /*! \name Internal functions */
   //! Sparse vector LUT creation
   int fill(int i=0, libbase::bitfield suffix="", int weight=-1);
   //! Watermark sequence creator
   void createsequence(const int tau);                      
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
   watermarkcode(const int n, const int k, const int s, const int N, const bool varyPs=true, const bool varyPd=true, const bool varyPi=true);
   ~watermarkcode() { free(); };
   // @}
   /*! \name Serialization Support */
   watermarkcode *clone() const { return new watermarkcode(*this); };
   const char* name() const { return shelper.name(); };
   // @}

   // Vector modem operations
   void modulate(const int N, const libbase::vector<int>& encoded, libbase::vector<bool>& tx);
   void demodulate(const channel<bool>& chan, const libbase::vector<bool>& rx, libbase::matrix<double>& ptable);

   // Informative functions
   int num_symbols() const { return 1<<k; };
   double energy() const { return n; };

   // Description & Serialization
   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

}; // end namespace

#endif

