/*!
 * \file
 *
 * Copyright (c) 2025 Aaron Abela.
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

#ifndef __qkd_errors_hamming_h
#define __qkd_errors_hamming_h

#include "config.h"
#include "experiment/results_collector.h"
#include "vector.h"
#include <string>
#include <sstream>

namespace libcomm
{

/*!
 * \brief   CommSys Results - Symbol/Frame Error Rates/SKR
 *
 *
 * Implements standard error rate calculators and SKR for CV-QKD
 * with Gaussian modulated coherent states.
 */
class qkd_errors_hamming :  public results_collector<typename libbase::vector<bool>>
{
protected:
    /*! \name System Interface */
    int source_length = 0; //<! Number of generated qubits per frame
    int secret_key_length = 0; //<! Calculated length of secret key from parameter estimation
public:
    virtual ~qkd_errors_hamming() {}

    /*! \name Public interface */
    void init(const queryable& system) override;

    void compute_result_and_accumulate(libbase::vector<double>& accumulated_result,
                       libbase::vector<uint64_t>& accumulated_count,
                       const libbase::vector<bool>& key_KA,
                       const libbase::vector<bool>& key_KB) const override;

    /*! \copydoc experiment::count()
     * We count the number of symbol, frame errors and secret key rate for
     * CV-QKD.
     */
    int result_count() const
    {
        return 4;
    } // Accounts for the current results in updateresults().

    /*! \copydoc experiment::result_description()
     *
     * The description is a string which indicates symbol or
     * frame error rates or SKR.
     */
    std::string result_description(int i) const override
    {
        assert(i >= 0 && i < result_count());
        switch (i) {
        case 0:
            return "SKR";
        case 1:
            return "AKR";
        case 2:
            return "SER";
        case 3:
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
        return "Secret Key Rate and Symbol/Frame Error Rates";
    }

    // Serialization Support
    DECLARE_SERIALIZER(qkd_errors_hamming)
};

} // namespace libcomm

#endif
