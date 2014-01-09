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

#ifndef __sparse_h
#define __sparse_h

#include "config.h"
#include "vector.h"
#include "bitfield.h"

namespace libbase {

/*!
 * \brief   Sparse Codebook Class.
 * \author  Johann Briffa
 *
 * Defines a codebook with the 'q' lowest-weight codewords of length 'n'.
 */

class sparse {
private:
   /*! \name Object representation */
   int n; //<! Codeword length
   vector<int> lut; //<! Table to hold codeword values
   // @}

private:
   /*! \name Internal functions */
   int fill(int i, bitfield suffix, int weight);
   // @}

public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   sparse()
      {
      }
   //! Principal constructor
   sparse(const int q, const int n)
      {
      init(q, n);
      }
   //! Virtual destructor
   virtual ~sparse()
      {
      }
   // @}

   /*! \name Codebook interface */
   /*! \brief Codebook setup function
    * \param q Number of codewords
    * \param n Length of each codeword in bits
    *
    * Creates a codebook with the 'q' lowest-weight codewords of length 'n'.
    */
   void init(const int q, const int n);
   //! Return indexed value
   int operator()(const int i) const
      {
      return lut(i);
      }
   //! Return whole codebook
   operator vector<int>() const
      {
      return lut;
      }
   // @}

   /*! \name Informative functions */
   //! The size of the codebook
   int size() const
      {
      return lut.size();
      }
   // @}
};

} // end namespace

#endif
