#ifndef __commsys_h
#define __commsys_h

#include "config.h"
#include "experiment.h"
#include "randgen.h"
#include "codec.h"
#include "mapper.h"
#include "modulator.h"
#include "puncture.h"
#include "channel.h"
#include "serializer.h"

namespace libcomm {

/*!
   \brief   CommSys Results - Bit/Symbol/Frame Error Rates.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (19 Feb 2008)
   - Moved standard error rate calculators into this class
   - Added get_alphabetsize()
*/
class commsys_errorrates {
protected:
   /*! \name System Interface */
   //! The number of decoding iterations performed
   virtual int get_iter() const = 0;
   //! The number of information symbols per block
   virtual int get_symbolsperblock() const = 0;
   //! The information symbol alphabet size
   virtual int get_alphabetsize() const = 0;
   //! The number of bits per information symbol
   virtual int get_bitspersymbol() const = 0;
   // @}
   /*! \name Helper functions */
   int countbiterrors(const libbase::vector<int>& source, const libbase::vector<int>& decoded) const;
   int countsymerrors(const libbase::vector<int>& source, const libbase::vector<int>& decoded) const;
   // @}
public:
   virtual ~commsys_errorrates() {};
   /*! \name Public interface */
   void updateresults(libbase::vector<double>& result, const int i, const libbase::vector<int>& source, const libbase::vector<int>& decoded) const;
   /*! \copydoc experiment::count()
       For each iteration, we count the number of symbol and frame errors
   */
   int count() const { return 2*get_iter(); };
   int get_multiplicity(int i) const;
   std::string result_description(int i) const;
   // @}
};

/*!
   \brief   Common Base for Communication Systems.
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
        and placed them instead within transmitandreceive(); this causes less than
        4% performance penalty, even on small (and therefore fast) codes.
      - Removed also source/decoded, refactoring internal functions as necessary.
      - Extracted result-updating for accumulation of BER/SER/FER as a separate
        function, and made this virtual rather than cycleonce(); this facilitates
        derivation of the class for the purposes of collecting different result
        sets.
   - Abstracted commsys:
      - Common elements, consisting of source, codec, and channel, created in this
        base templated class basic_commsys.

   \version 2.10 (28 Jan 2008)
   - Moved modulator object to the common class from the sigspace specialization;
     the order of serialization is now changed back to what it used to be, where
     the channel goes first, followed by the modulator, and finally the coded.

   \version 2.20 (19 Feb 2008)
   - Moved standard error rate calculators into separate class
   - Result set calculation now included as a template parameter
   - Default result set is commsys_errorrates
   - Added get_alphabetsize()

   \version 2.21 (22 Feb 2008)
   - Added seeding for modulator block

   \version 2.30 (25 Feb 2008)
   - Added mapper block between codec and modulator
*/

template <class S, class R=commsys_errorrates> class basic_commsys
   : public experiment_binomial, public R {
protected:
   /*! \name Bound objects */
   //! Flag to indicate whether the objects should be released on destruction
   bool  internallyallocated;
   libbase::randgen     *src;    //!< Source data sequence generator
   codec                *cdc;    //!< Error-control codec
   mapper               *map;    //!< Symbol-mapper (encoded output to transmitted symbols)
   modulator<S>         *modem;  //!< Modulation scheme
   channel<S>           *chan;   //!< Channel model
   // @}
   /*! \name Computed parameters */
   int  tau;   //!< Codec block size (in time-steps)
   int  m;     //!< Tail length required by codec (may be zero)
   int  N;     //!< Alphabet size for encoder output
   int  K;     //!< Alphabet size for source data
   int  k;     //!< Bit width for source data symbols (\f$ K = 2^k \f$)
   int  iter;  //!< Number of iterations the decoder will do
   // @}
protected:
   /*! \name Setup functions */
   void init();
   void clear();
   void free();
   // @}
   /*! \name Internal functions */
   libbase::vector<int> createsource();
   //! Perform a complete transmit/receive cycle, except for final decoding
   virtual void transmitandreceive(libbase::vector<int>& source) = 0;
   void cycleonce(libbase::vector<double>& result);
   // @}
   /*! \name System Interface for Results */
   int get_iter() const { return iter; };
   int get_symbolsperblock() const { return tau-m; };
   int get_alphabetsize() const { return K; };
   int get_bitspersymbol() const { return k; };
   // @}
public:
   /*! \name Constructors / Destructors */
   basic_commsys(libbase::randgen *src, codec *cdc, mapper *map, modulator<S> *modem, channel<S> *chan);
   basic_commsys(const basic_commsys<S,R>& c);
   basic_commsys() { clear(); };
   virtual ~basic_commsys() { free(); };
   // @}

   // Experiment parameter handling
   void seedfrom(libbase::random& r);
   void set_parameter(double x) { chan->set_parameter(x); };
   double get_parameter() { return chan->get_parameter(); };

   // Experiment handling
   void sample(libbase::vector<double>& result);
   int count() const { return R::count(); };
   int get_multiplicity(int i) const { return R::get_multiplicity(i); };
   std::string result_description(int i) const { return R::result_description(i); };

   /*! \name Component object handles */
   //! Get error-control codec
   const codec *getcodec() const { return cdc; };
   //! Get symbol mapper
   const mapper *getmapper() const { return map; };
   //! Get modulation scheme
   const modulator<S> *getmodem() const { return modem; };
   //! Get channel model
   const channel<S> *getchan() const { return chan; };
   // @}

   // Description & Serialization
   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

/*!
   \brief   Base Communication System.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (25 Jan 2008)
   - Abstracted commsys:
      - General templated commsys derived from generic base; this cannot be
        instantiated, as it is still abstract.

   \version 2.00 (12 Feb 2008)
   - Integrated functionality of binary variant into this class.
   - This class therefore can now be instantiated.
   - An explicit instantiation for bool is present to replace the functionality
     of the earlier specific specialization.
   - Added explicit instantiations for gf types.

   \version 2.10 (19 Feb 2008)
   - Added result set calculation as a template parameter
   - Default result set is commsys_errorrates
   - Added explicit realizations of hist_symerr and prof_pos variants for bool channel
   - Added explicit realization of prof_sym variant for bool channel
*/
template <class S, class R=commsys_errorrates> class commsys : public basic_commsys<S,R> {
protected:
   /*! \name Internal functions */
   void transmitandreceive(libbase::vector<int>& source);
   // @}
public:
   // Serialization Support
   DECLARE_SERIALIZER(commsys)
};

/*!
   \brief   Signal-Space Communication System.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (25 Jan 2008)
   - Abstracted commsys:
      - This explicit specialization for sigspace channel contains objects
        and functions remaining from the templated base, and is equivalent
        to the old commsys class; anything that used to use 'commsys' can
        now use this specialization.

   \version 1.01 (28 Jan 2008)
   - Changed reference from modulator to modulator<sigspace>

   \version 1.10 (18 Apr 2008)
   - Implemented serialization of puncturing system; the canonical form
     requires the addition of a 'false' flag at the end of the stream to signal
     that there is no puncturing. In order not to break current input files, the
     flag is assumed to be false (with no error) if we have reached the end of the
     stream.
*/
template <class R> class commsys<sigspace,R> : public basic_commsys<sigspace,R> {
protected:
   /*! \name Bound objects */
   puncture             *punc;   //!< Puncturing (operates on signal-space symbols)
   // @}
protected:
   /*! \name Setup functions */
   void init();
   void clear();
   void free();
   // @}
   /*! \name Internal functions */
   void transmitandreceive(libbase::vector<int>& source);
   // @}
public:
   /*! \name Constructors / Destructors */
   commsys<sigspace,R>(libbase::randgen *src, codec *cdc, mapper *map, modulator<sigspace> *modem, puncture *punc, channel<sigspace> *chan);
   commsys<sigspace,R>(const commsys<sigspace,R>& c);
   commsys<sigspace,R>() { clear(); };
   virtual ~commsys<sigspace,R>() { free(); };
   // @}

   /*! \name Component object handles */
   //! Get puncturing scheme
   const puncture *getpunc() const { return punc; };
   // @}

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(commsys)
};

}; // end namespace

#endif
