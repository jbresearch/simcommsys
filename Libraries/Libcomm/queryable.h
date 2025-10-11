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

#ifndef __queryable_h
#define __queryable_h

#include "config.h"

#include <any>

namespace libcomm
{

/*!
 * \brief   Queryable Class Interface.
 * \author  Johann Briffa
 *
 * Defines a class that can be queried for specific values.
 */

class queryable
{
public:
    /*! \name Constructors / Destructors */
    virtual ~queryable() {}
    // @}

    /*! \name Query handling */
    /*!
     * \brief Getter for values by index
     * \param[in] index A system-dependent index into the dictionary of
     * values that can be obtained
     * \return The requested value
     * 
     * This method is used by the results collector interface, so that a
     * specific results collector can obtain relevant values from the
     * system being simulated.
     */
    virtual std::any get_value(const int index) const = 0;
    // @}
};

} // namespace libcomm

#endif
