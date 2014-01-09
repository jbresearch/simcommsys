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

#ifndef __lsb_h
#define __lsb_h

#include "config.h"
#include "embedder.h"
#include "randgen.h"

namespace libcomm {

/*!
 * \brief   LSB Embedder/Extractor.
 * \author  Johann Briffa
 *
 * This class implements LSB embedding that can be applied to any integer type.
 * Two types of embedding are implemented:
 * - LSB replacement (using modulo arithmetic)
 * - LSB matching (or ±1 embedding)
 */

template <class S>
class lsb : public embedder<S> {
private:
   /*! \name User-defined parameters */
   int M; //! Alphabet size in symbols
   enum al_enum {
      AL_REPLACEMENT, //!< LSB replacement
      AL_MATCHING, //!< LSB matching
      AL_UNDEFINED
   } algorithm;
   // @}
   /*! \name Internal representation */
   mutable libbase::randgen r; //!< ± selector for matching
   // @}
protected:
   //! Verifies that object is in a valid state
   void test_invariant() const
      {
      assert(M >= 2);
      }
public:
   lsb(const int M = 2) :
      M(M), algorithm(AL_REPLACEMENT)
      {
      }

   // Setup functions
   void seedfrom(libbase::random& r)
      {
      libbase::int32u seed = r.ival();
      this->r.seed(seed);
      }

   // Atomic embedder operations
   const S embed(const int i, const S s) const
      {
      assert(i >= 0 && i < M);
      switch (algorithm)
         {
         case AL_REPLACEMENT:
            return S(s - (s % M) + i);
         case AL_MATCHING:
            {
            int delta = i - (s % M);
            // if the LSB is correct, leave as is
            if (delta == 0)
               return s;
            // otherwise randomly decided to add/subtract
            if (r.ival(2) == 0)
               delta = (delta > 0) ? delta - M : M - delta;
            // return result (does not respect representation limits)
            return S(s + delta);
            }
         default:
            failwith("Unknown algorithm");
            return s;
         }
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
