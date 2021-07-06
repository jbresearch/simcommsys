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

#ifndef __source_sequential_h
#define __source_sequential_h

#include "config.h"
#include "serializer.h"
#include "source.h"
#include <sstream>

namespace libcomm
{

/*!
 * \brief   All-sequential source.
 * \author  Johann Briffa
 *
 * Implements a trivial source that always returns the next symbol from a
 * given repeating sequence.
 */

template <class S, template <class> class C = libbase::vector>
class sequential : public source<S, C>
{
private:
    /*! \name Internal representation */
    libbase::vector<S> input_vectors; //!< user sequence of input symbols
    int index;                        //!< index of next element to output
                                      // @}
public:
    //! Default constructor
    sequential() : index(0) {}
    //! Main constructor
    sequential(libbase::vector<S> input_vectors)
        : input_vectors(input_vectors), index(0)
    {
    }

    //! Generate a single source element
    S generate_single()
    {
        S value = input_vectors(index);
        index = (index + 1) % input_vectors.size();
        return value;
    }

    //! Description
    std::string description() const
    {
        std::ostringstream sout;
        sout << "Sequential source [" << input_vectors.size() << " elements]";
        return sout.str();
    }

    // Serialization Support
    DECLARE_SERIALIZER(sequential)
};

} // namespace libcomm

#endif
