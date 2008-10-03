#ifndef __commsys_h
#define __commsys_h

#include "config.h"
#include "randgen.h"
#include "codec.h"
#include "mapper.h"
#include "blockmodem.h"
#include "channel.h"
#include "serializer.h"

namespace libcomm {

/*!
   \brief   Common Base for Communication System.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   General templated commsys.
   - Integrates functionality of binary variant.
   - Explicit instantiations for bool and gf types are present.
*/

template <class S>
class basic_commsys {
protected:
   /*! \name Bound objects */
   //! Flag to indicate whether the objects should be released on destruction
   bool  internallyallocated;
   codec          *cdc;    //!< Error-control codec
   mapper         *map;    //!< Symbol-mapper (encoded output to transmitted symbols)
   blockmodem<S>   *mdm;  //!< Modulation scheme
   channel<S>     *chan;   //!< Channel model
   // @}
   /*! \name Computed parameters */
   int  M;     //!< Alphabet size for modulation symbols
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
public:
   /*! \name Constructors / Destructors */
   basic_commsys(codec *cdc, mapper *map, blockmodem<S> *mdm, channel<S> *chan);
   basic_commsys(const basic_commsys<S>& c);
   basic_commsys() { clear(); };
   virtual ~basic_commsys() { free(); };
   // @}

   /*! \name Communication System Setup */
   virtual void seedfrom(libbase::random& r);
   //! Get error-control codec
   codec *getcodec() const { return cdc; };
   //! Get symbol mapper
   mapper *getmapper() const { return map; };
   //! Get modulation scheme
   blockmodem<S> *getmodem() const { return mdm; };
   //! Get channel model
   channel<S> *getchan() const { return chan; };
   // @}

   /*! \name Communication System Interface */
   //! Perform complete encode path (encode -> map -> modulate)
   libbase::vector<S> encode(const libbase::vector<int>& source);
   //! Perform complete translation path (demodulate -> unmap -> translate)
   void translate(const libbase::vector<S>& received);
   //! Perform a complete transmit/receive cycle, except for final decoding
   virtual void transmitandreceive(const libbase::vector<int>& source);
   // @}

   /*! \name Informative functions */
   //! Overall mapper rate
   double rate() const { return cdc->rate() * map->rate(); };
   //! Input (ie. source/decoded) block size in symbols
   int input_block_size() const { return cdc->input_block_size(); };
   //! Output (ie. transmitted/received) block size in symbols
   int output_block_size() const { return map->determine_output_block_size(cdc->output_block_size()); };
   // @}

   // Description
   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

/*!
   \brief   General Communication System.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   General templated commsys, directly derived from common base.
*/

template <class S>
class commsys : public basic_commsys<S> {
public:
   // Serialization Support
   DECLARE_CONCRETE_BASE_SERIALIZER(commsys)
};

/*!
   \brief   Signal-Space Communication System.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   This explicit specialization for sigspace channel contains objects and
   functions remaining from the templated base, and is generally equivalent
   to the old commsys class; anything that used to use 'commsys' can now use
   this specialization.

   \note Support for puncturing has changed from its previous operation in
         signal-space to the more general mapper layer.

   \note Serialization of puncturing system is implemented; the canonical
         form this requires the addition of a 'false' flag at the end of the
         stream to signal that there is no puncturing. In order not to break
         current input files, the flag is assumed to be false (with no error)
         if we have reached the end of the stream.
*/
template <>
class commsys<sigspace> : public basic_commsys<sigspace> {
protected:
   /*! \name Setup functions */
   void init();
   // @}
public:
   // Serialization Support
   DECLARE_CONCRETE_BASE_SERIALIZER(commsys)
};

}; // end namespace

#endif
