#ifndef __embedder_h
#define __embedder_h

#include "config.h"
#include "random.h"
#include "serializer.h"
#include <iostream>
#include <string>

namespace libcomm {

/*!
 * \brief   Common Data Embedder/Extractor Interface.
 * \author  Johann Briffa
 *
 * \par Version Control:
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Class defines common interface for embedder classes.
 */

template <class S>
class basic_embedder {
public:
   /*! \name Constructors / Destructors */
   //! Virtual destructor
   virtual ~basic_embedder()
      {
      }
   // @}

   /*! \name Atomic embedder operations */
   /*!
    * \brief Embed a single symbol
    * \param   data Index into the symbol alphabet (data to embed)
    * \param   host Host value into which to embed data
    * \return  Stego-value, encoding the given data
    */
   virtual const S embed(const int data, const S host) const = 0;
   /*!
    * \brief Extract a single symbol
    * \param   rx Received (possibly corrupted) stego-value
    * \return  Index corresponding to most-likely transmitted symbol
    */
   virtual const int extract(const S& rx) const = 0;
   // @}

   /*! \name Setup functions */
   //! Seeds any random generators from a pseudo-random sequence
   virtual void seedfrom(libbase::random& r)
      {
      }
   // @}

   /*! \name Informative functions */
   //! Symbol alphabet size at input
   virtual int num_symbols() const = 0;
   // @}

   /*! \name Description */
   //! Description output
   virtual std::string description() const = 0;
   // @}
};

/*!
 * \brief   Data Embedder/Extractor Base.
 * \author  Johann Briffa
 *
 * \par Version Control:
 * - $Revision$
 * - $Date$
 * - $Author$
 */

template <class S>
class embedder : public basic_embedder<S> , public libbase::serializable {
public:
   // Serialization Support
DECLARE_BASE_SERIALIZER( embedder)
};

} // end namespace

#endif
