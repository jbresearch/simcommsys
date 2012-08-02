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

#ifndef __channel_insdel_h
#define __channel_insdel_h

#include "channel.h"

namespace libcomm {

/*!
 * \brief   Insertion-Deletion Channel Interface.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Defines the additional interface methods for insertion-deletion channels.
 */

template <class S>
class channel_insdel : public channel<S> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<int> array1i_t;
   // @}
public:
   /*! \name Insertion-deletion channel functions */
   /*!
     * \brief Get the actual channel drift at time 't' of last transmitted frame.
     */
   virtual int get_drift(int t) const = 0;
   /*!
     * \brief Get the actual channel drift at a set of times 't' of last transmitted frame.
     */
   virtual array1i_t get_drift(const array1i_t& t) const
       {
       // allocate space for results
       array1i_t result(t.size());
       // consider each time index in the order given
       for (int i = 0; i < t.size(); i++)
          result(i) = get_drift(t(i));
       return result;
       }
   // @}
};

} // end namespace

#endif
