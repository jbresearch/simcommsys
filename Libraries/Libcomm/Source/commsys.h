#ifndef __commsys_h
#define __commsys_h

#include "config.h"
#include "experiment.h"
#include "randgen.h"
#include "codec.h"
#include "modulator.h"
#include "puncture.h"
#include "channel.h"
#include "serializer.h"

namespace libcomm {

/*!
   \brief   Communication System.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.10 (7 Jun 1999)
   modified the system to comply with codec 1.10.

   \version 1.20 (30 Jul 1999)
   added option to speed up Turbo decoding (by stopping when an iteration does not
   improve the error rate).

   \version 1.21 (26 Aug 1999)
   modified stopping criterion for samples such that sample granularity is just above 0.5s
   based on a timer rather than on the number of symbols transmitted

   \version 1.30 (2 Sep 1999)
   added a hook for clients to know the number of frames simulated in a particular run.

   \version 1.31 (1 Mar 2002)
   edited the classes to be compileable with Microsoft extensions enabled - in practice,
   the major change is in for() loops, where MS defines scope differently from ANSI.
   Rather than taking the loop variables into function scope, we chose to avoid having
   more than one loop per function, by defining private helper functions (or doing away
   with them if there are better ways of doing the same operation).

   \version 1.32 (6 Mar 2002)
   changed vcs version variable from a global to a static class variable.
   also changed use of iostream from global to std namespace.

   \version 1.40 (18 Mar 2002)
   changed constructor to take also the modem and an optional puncturing system, besides
   the already present random source (for generating the source stream), the channel
   model, and the codec. This change was necessitated by the definition of codec 1.41.
   Also removed the extremely hazardous "fast" option for speed enhancement (see discussion
   in my journal last Fall). Finally, changed the sample loop to bail out after 0.5s
   rather than after at least 1000 modulation symbols have been transmitted (records
   above indicate this should have been done already - why not?).

   \version 1.41 (19 Mar 2002)
   changed the class definition a little to make it more easily used as a base class for
   any type of communication system simulation (essentially after noticing that all
   other comm simulation class had several functions that were essentially identical).
   This involved changing the access to the createsource and cycleonce from private to
   protected, and also adding an extra function transmitandreceive() which does the
   common work from encoding through demodulation. Finally, the cycleonce() function is
   now virtual - this is the function that derived classes need to override in order to
   make full use of the common commsys framework. Also, the base experiment is now a
   public base not a public virtual one. Also made data members protected to allow
   easier derivation. Also fixed a bug (that we were not setting the channel Eb anywhere;
   we are now doing that in the constructor).

   \version 1.42 (24 Mar 2002)
   fixed a bug in commsys constructor - was calling punc->rate() even when punc was NULL;
   now we just use the codec rate when punc is undefined.

   \version 1.43 (17 Jul 2006)
   in constructor, made explicit conversion of round's output to int.

   \version 1.44 (25 Jul 2006)
   in transmitandreceive(), moved the modulation line outside the punc decision. Should
   not have any effect on results or speed.

   \version 1.50 (30 Oct 2006)
   - defined class and associated data within "libcomm" namespace.
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.

   \version 1.60 (24 Apr 2007)
   - added serialization facility requirement.
   - added copy constructor

   \version 1.61 (29 Oct 2007)
   - updated clone() to return this object's type, rather than its base class type. [cf. Stroustrup 15.6.2]

   \version 1.62 (29 Nov 2007)
   - added getters for internal objects

   \version 1.70 (18 Dec 2007)
   - Modified sample() as in experiment 1.40

   \version 1.80 (19 Dec 2007)
   - Added computation and return of symbol error rate (always performed, even if
     symbols are binary and therefore SER=BER).

   \version 1.81 (20 Dec 2007)
   - Fixed memory leak, where dynamically allocated objects were not being deleted
     on object destruction.
   - Cleaned up refactoring work on SER computation

   \version 1.82 (22 Jan 2008)
   - Modified serialization method, adopting the format used in simcommsys 1.26;
     note that this does not support puncturing. This format was adopted in favor
     of changing the file format used in current simulations because:
      - we do not wish to break the format unnecessarily
      - support for puncturing needs to change anyway, from its current operation
        in signal-space to a more general mapper layer
   - Made default constructor public, to allow direct serialization, rather than
     just through 'experiment'.
   - Modified seed() so that different (but still deterministic seeds are used in
     the various components; this ensures that the sequences do not just 'happen'
     to be the same.

   \version 1.83 (24 Jan 2008)
   - Changed reference from channel to channel<sigspace>

   \version 2.00 (25 Jan 2008)
   - In order to facilitate abstraction of commsys beyond use on sigspace channels:
      - Removed most working variables (except source/decoded) from memebers,
        and placed them instead within transmitandreceive().
*/

class commsys : public experiment {
   /*! \name Serialization */
   static const libbase::serializer shelper;
   static void* create() { return new commsys; };
   // @}
protected:
   /*! \name Bound objects */
   //! Flag to indicate whether the objects should be released on destruction
   bool  internallyallocated;
   libbase::randgen     *src;    //!< Source data sequence generator
   codec                *cdc;    //!< Error-control codec
   modulator            *modem;  //!< Modulation scheme
   puncture             *punc;   //!< Puncturing (operates on signal-space symbols)
   channel<sigspace>    *chan;   //!< Channel model
   // @}
   /*! \name Computed parameters */
   int  tau;   //!< Codec block size (in time-steps)
   int  m;     //!< Tail length required by codec (may be zero)
   int  N;     //!< Alphabet size for encoder output
   int  K;     //!< Alphabet size for source data
   int  k;     //!< Bit width for source data symbols (\f$ K = 2^k \f$)
   int  iter;  //!< Number of iterations the decoder will do
   // @}
   /*! \name Working variables */
   libbase::vector<int>       source;  //!< Sequence of source data symbols
   libbase::vector<int>       decoded; //!< Sequence of decoded symbols
   // @}
protected:
   /*! \name Setup functions */
   void init();
   void clear();
   void free();
   // @}
   /*! \name Internal functions */
   void createsource();
   void transmitandreceive();
   int countbiterrors() const;
   int countsymerrors();
   virtual void cycleonce(libbase::vector<double>& result);
   // @}
public:
   /*! \name Constructors / Destructors */
   commsys(libbase::randgen *src, codec *cdc, modulator *modem, puncture *punc, channel<sigspace> *chan);
   commsys(const commsys& c);
   commsys() { clear(); };
   ~commsys() { free(); };
   // @}

   //*! \name Serialization Support */
   commsys *clone() const { return new commsys(*this); };
   const char* name() const { return shelper.name(); };
   // @}

   // Experiment parameter handling
   void seed(int s);
   void set_parameter(double x) { chan->set_parameter(x); };
   double get_parameter() { return chan->get_parameter(); };

   // Experiment handling
   void sample(libbase::vector<double>& result);
   int count() const { return 3*iter; };

   /*! \name Component object handles */
   //! Get error-control codec
   const codec *getcodec() const { return cdc; };
   //! Get modulation scheme
   const modulator *getmodem() const { return modem; };
   //! Get puncturing scheme
   const puncture *getpunc() const { return punc; };
   //! Get channel model
   const channel<sigspace> *getchan() const { return chan; };
   // @}

   // Description & Serialization
   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

}; // end namespace

#endif
