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

#ifndef __prof_sym_h
#define __prof_sym_h

#include "config.h"
#include "errors_hamming.h"
#include <sstream>

namespace libcomm
{

/*!
 * \brief   CommSys Results - Symbol-Value Error Profile.
 * \author  Johann Briffa
 *
 * Computes symbol-error histogram as dependent on source symbol value.
 */

class prof_sym : public errors_hamming
{
protected:
    /*! \name System parameters */
    //! The information symbol alphabet size
    int alphabetsize;
    // @}
public:
    prof_sym() : alphabetsize(0) {}
    /*! \name Results collector interface */
    void init(const queryable& system) override;
    void compute_result_and_accumulate(
        libbase::vector<double>& accumulated_result,
        libbase::vector<uint64_t>& accumulated_count,
        const libbase::vector<int>& source,
        const libbase::vector<int>& decoded) const override;
    /*! \copydoc experiment::result_count()
     * We count the number of symbol errors for every input alphabet symbol
     * value.
     */
    int result_count() const override { return alphabetsize; }
    /*! \copydoc experiment::result_description()
     *
     * The description is a string SER_X, where 'X' is the symbol value
     * (starting at zero).
     */
    std::string result_description(int i) const override
    {
        assert(i >= 0 && i < result_count());
        std::ostringstream sout;
        sout << "SER_" << i;
        return sout.str();
    }
    // @}

    // Description
    std::string description() const override
    {
        return "Symbol-Value Error Profile";
    }

    // Serialization Support
    DECLARE_SERIALIZER(prof_sym)
};

} // namespace libcomm

#endif
