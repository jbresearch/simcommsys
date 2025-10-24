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

#ifndef __errors_hamming_h
#define __errors_hamming_h

#include "config.h"
#include "experiment/results_collector.h"

#include <sstream>
#include <stdexcept>

namespace libcomm
{

/*!
 * \brief   CommSys Results - Symbol/Frame Error Rates.
 * \author  Johann Briffa
 *
 * Implements standard error rate calculators.
 */
class errors_hamming : public results_collector<typename libbase::vector<int>>
{
protected:
    /*! \name System parameters */
    //! The number of information symbols per block
    int symbolsperblock;
    // @}
public:
    errors_hamming() : symbolsperblock(0) {}
    virtual ~errors_hamming() {}
    /*! \name Results collector interface */
    void init(const queryable& system) override;
    void compute_result_and_accumulate(
        libbase::vector<double>& accumulated_result,
        libbase::vector<uint64_t>& accumulated_count,
        const libbase::vector<int>& source,
        const libbase::vector<int>& decoded) const override;
    /*! \copydoc experiment::result_count()
     * We count the number of symbol and frame errors
     */
    int result_count() const override { return 2; }
    /*! \copydoc experiment::result_description()
     *
     * The description is SER or FER to indicate symbol or frame error rate
     * respectively.
     */
    std::string result_description(int i) const override
    {
        assert(i >= 0 && i < result_count());
        switch (i) {
        case 0:
            return "SER";
        case 1:
            return "FER";
        }
        // This should never happen
        std::ostringstream sout;
        sout << "Index " << i << " out of range. Valid range is [0,"
             << result_count() - 1 << "].";
        throw std::out_of_range(sout.str());
    }
    // @}

    // Description
    std::string description() const override
    {
        return "Symbol/Frame Error Rates";
    }

    // Serialization Support
    DECLARE_SERIALIZER(errors_hamming)
};

} // namespace libcomm

#endif
