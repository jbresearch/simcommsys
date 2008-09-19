#ifndef __commsys_h
#define __commsys_h

#include "config.h"
#include "commsys_errorrates.h"
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
   \brief   Common Base for Communication Systems.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \todo Support for puncturing needs to change from its current operation
         in signal-space to a more general mapper layer
*/

template <class S, class R=commsys_errorrates>
class basic_commsys_simulator
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
   int  M;     //!< Alphabet size for modulation symbols
   int  N;     //!< Alphabet size for encoder output
   int  K;     //!< Alphabet size for source data
   int  k;     //!< Bit width for source data symbols (\f$ K = 2^k \f$)
   int  iter;  //!< Number of iterations the decoder will do
   // @}
   /*! \name Internal state */
   libbase::vector<int> last_event;
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
   basic_commsys_simulator(libbase::randgen *src, codec *cdc, mapper *map, modulator<S> *modem, channel<S> *chan);
   basic_commsys_simulator(const basic_commsys_simulator<S,R>& c);
   basic_commsys_simulator() { clear(); };
   virtual ~basic_commsys_simulator() { free(); };
   // @}

   // Experiment parameter handling
   void seedfrom(libbase::random& r);
   void set_parameter(const double x) { chan->set_parameter(x); };
   double get_parameter() const { return chan->get_parameter(); };

   // Experiment handling
   void sample(libbase::vector<double>& result);
   int count() const { return R::count(); };
   int get_multiplicity(int i) const { return R::get_multiplicity(i); };
   std::string result_description(int i) const { return R::result_description(i); };
   libbase::vector<int> get_event() const { return last_event; };

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

   General templated commsys_simulator derived from generic base.
   - Integrates functionality of binary variant.
   - Explicit instantiations for bool and gf types are present.
*/
template <class S, class R=commsys_errorrates>
class commsys_simulator : public basic_commsys_simulator<S,R> {
protected:
   /*! \name Internal functions */
   void transmitandreceive(libbase::vector<int>& source);
   // @}
public:
   // Serialization Support
   DECLARE_SERIALIZER(commsys_simulator)
};

/*!
   \brief   Signal-Space Communication System.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   This explicit specialization for sigspace channel contains objects and
   functions remaining from the templated base, and is equivalent to the
   old commsys_simulator class; anything that used to use 'commsys_simulator' can now use this
   specialization.

   \note Serialization of puncturing system is implemented; the canonical
         form this requires the addition of a 'false' flag at the end of the
         stream to signal that there is no puncturing. In order not to break
         current input files, the flag is assumed to be false (with no error)
         if we have reached the end of the stream.
*/
template <class R>
class commsys_simulator<sigspace,R> : public basic_commsys_simulator<sigspace,R> {
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
   commsys_simulator<sigspace,R>(libbase::randgen *src, codec *cdc, mapper *map, modulator<sigspace> *modem, puncture *punc, channel<sigspace> *chan);
   commsys_simulator<sigspace,R>(const commsys_simulator<sigspace,R>& c);
   commsys_simulator<sigspace,R>() { clear(); };
   virtual ~commsys_simulator<sigspace,R>() { free(); };
   // @}

   /*! \name Component object handles */
   //! Get puncturing scheme
   const puncture *getpunc() const { return punc; };
   // @}

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(commsys_simulator)
};

}; // end namespace

#endif
