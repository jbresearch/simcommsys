/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __blockprocess_h
#define __blockprocess_h

#include "config.h"

namespace libcomm {

/*!
 * \brief   Block-Processed Interface.
 * \author  Johann Briffa
 *
 * This class defines the interface for an object that gets processed in a
 * blockwise fashion.
 */

class blockprocess {
protected:
   /*! \name Internal representation */
   mutable bool dirty; //!< Flag indicating this block has been processed
   // @}

protected:
   /*! \name Interface with derived classes */
   //! Sets up object for the next block
   virtual void advance() const
      {
      }
   //! Hook for processes when status changes
   virtual void status_changed() const
      {
      }
   // @}

public:
   /*! \name Constructors / Destructors */
   blockprocess() : dirty(true)
      {
      }
   virtual ~blockprocess()
      {
      }
   // @}

   /*! \name Block-processing operations */
   //! Always advance to the next block
   void advance_always() const
      {
      advance();
      mark_as_clean();
      }
   //! Advance to the next block only if this block is 'dirty'
   void advance_if_dirty() const
      {
      if (dirty)
         {
         advance();
         mark_as_clean();
         }
      }
   //! Mark this block as 'dirty'
   void mark_as_dirty() const
      {
      dirty = true;
      status_changed();
      }
   //! Mark this block as not 'dirty'
   void mark_as_clean() const
      {
      dirty = false;
      status_changed();
      }
   // @}
};

} // end namespace

#endif
