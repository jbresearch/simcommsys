#ifndef __blockprocess_h
#define __blockprocess_h

#include "config.h"

namespace libcomm {

/*!
   \brief   Block-Processed Interface.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   This class defines the interface for an object that gets processed in a
   blockwise fashion.
*/

class blockprocess {
protected:
   /*! \name Internal representation */
   mutable bool advanced;   //!< Flag indicating object is ready for next block
   // @}

protected:
   /*! \name Interface with derived classes */
   //! Sets up object for the next block
   virtual void advance() const {};
   // @}

public:
   /*! \name Constructors / Destructors */
   blockprocess() { advanced = false; };
   virtual ~blockprocess() {};
   // @}

   /*! \name Block-processing operations */
   //! Always advance to the next block
   void advance_always() const;
   //! Advance to the next block only if this block is 'dirty'
   void advance_if_dirty() const;
   //! Mark this block as 'dirty'
   void mark_as_dirty() const { advanced = false; };
   // @}
};

}; // end namespace

#endif
