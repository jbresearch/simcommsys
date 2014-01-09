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

#ifndef __randperm_h
#define __randperm_h

#include "config.h"
#include "random.h"
#include "vector.h"

namespace libbase {

/*!
 * \brief   Random Permutation Class.
 * \author  Johann Briffa
 *
 * Defines a random permutation of the set {0,1,..N-1} for given size N.
 */

class randperm {
private:
   /*! \name Object representation */
   //! Table to hold permutation values
   vector<int> lut;
   // @}

public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   randperm()
      {
      }
   //! Principal constructor
   randperm(const int N, random& r)
      {
      init(N, r);
      }
   //! Virtual destructor
   virtual ~randperm()
      {
      }
   // @}

   /*! \name Random permutation interface */
   /*! \brief Permutation setup function
    * \param N Size of permutation
    * \param r Random generator to use in creating permutation
    *
    * Sets up a random permutation of the set {0,1,..N-1} for given size N.
    */
   void init(const int N, random& r);
   //! Return indexed value
   int operator()(const int i) const
      {
      return lut(i);
      }
   // @}

   /*! \name Informative functions */
   //! The size of the permutation
   int size() const
      {
      return lut.size();
      }
   // @}
};

} // end namespace

#endif
