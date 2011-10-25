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
 * 
 * \section svn Version Control
 * - $Id$
 */

#ifndef __lsb_h
#define __lsb_h

#include "config.h"
#include "embedder.h"

namespace libcomm {

/*!
 * \brief   LSB Replacement Embedder/Extractor.
 * \author  Johann Briffa
 *
 * \par Version Control:
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * LSB replacement is defined in terms of modulo arithmetic and can be
 * applied to any integer class.
 */

template <class S>
class lsb : public embedder<S> {
private:
   /*! \name User-defined parameters */
   int M; //! Alphabet size in symbols
   // @}
protected:
   //! Verifies that object is in a valid state
   void test_invariant() const
      {
      assert(M >= 2);
      }
public:
   lsb(const int M = 2) :
      M(M)
      {
      }

   // Atomic embedder operations
   const S embed(const int i, const S s) const
      {
      assert(i >=0 && i < M);
      return S(s - (s % M) + i);
      }
   const int extract(const S& rx) const
      {
      return int(rx % M);
      }

   // Informative functions
   int num_symbols() const
      {
      return M;
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER( lsb)
};

} // end namespace

#endif
