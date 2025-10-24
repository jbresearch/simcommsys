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

#ifndef __prof_burst_h
#define __prof_burst_h

#include "config.h"
#include "errors_hamming.h"

namespace libcomm
{

/*!
 * \brief   CommSys Results - Error Burstiness Profile.
 * \author  Johann Briffa
 *
 * Determines separately the error probabilities for:
 * the first symbol in a frame
 * a symbol following a correctly-decoded one
 * a symbol following an incorrectly-decoded one
 */

class prof_burst : public errors_hamming
{
public:
    /*! \name Results collector interface */
    void compute_result_and_accumulate(
        libbase::vector<double>& accumulated_result,
        libbase::vector<uint64_t>& accumulated_count,
        const libbase::vector<int>& source,
        const libbase::vector<int>& decoded) const override;
    /*! \copydoc experiment::result_count()
     * We count respectively the number symbol errors:
     * - in the first frame symbol
     * - in subsequent symbols:
     * - if the prior symbol was correct (ie. joint probability)
     * - if the prior symbol was in error
     * - in the prior symbol (required when applying Bayes' rule
     * to the above two counts)
     */
    int result_count() const override { return 4; }
    /*! \copydoc experiment::result_description()
     *
     * The description is a string indicating the probability represented.
     */
    std::string result_description(int i) const override
    {
        assert(i >= 0 && i < result_count());
        switch (i) {
        case 0:
            return "P[e0]";
        case 1:
            return "P[ei|correct]";
        case 2:
            return "P[ei|error]";
        case 3:
            return "P[ei-1]";
        }
        return ""; // This should never happen
    }
    // @}

    // Description
    std::string description() const override
    {
        return "Error Burstiness Profile";
    }

    // Serialization Support
    DECLARE_SERIALIZER(prof_burst)
};

} // namespace libcomm

#endif
