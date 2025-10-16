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

#ifndef __cv_qkd_errors_hamming_h
#define __cv_qkd_errors_hamming_h

#include "config.h"
#include "source/quantum_gaussian_source.h"
#include "vector.h"
#include <string>

namespace libcomm
{

/*!
 * \brief   CommSys Results - Symbol/Frame Error Rates/SKR
 *
 *
 * Implements standard error rate calculators and SKR for CV-QKD
 * with Gaussian modulated coherent states.
 */
class cv_qkd_errors_hamming
{
protected:
    /*! \name System Interface */
    //! The number of information symbols per block
    virtual int get_symbolsperblock() const = 0;
    //! The information symbol alphabet size
    virtual int get_alphabetsize() const = 0;
    // @}
public:
    virtual ~cv_qkd_errors_hamming() {}
    /*! \name Public interface */
    void updateresults(libbase::vector<double>& result,
                       libbase::vector<gaussian_state> source,
                       libbase::vector<bool>& key_KA,
                       libbase::vector<bool>& key_KB) const;
    /*! \copydoc experiment::count()
     * We count the number of symbol, frame errors and secret key rate for
     * CV-QKD.
     */
    int count() const
    {
        return 3;
    } // Accounts for the current results in updateresults().
    /*! \copydoc experiment::get_multiplicity()
     *
     * Since results are organized as (symbol,frame) error count, the
     * multiplicity is respectively the number of symbols and the number of
     * frames (=1) per sample.
     */
    int get_multiplicity(int i) const
    {
        return (i == 0) ? get_symbolsperblock() : 1;
        assert(i >= 0 && i < count());
        switch (i) {
        case 0: // SKR
            return get_symbolsperblock();
        case 1: // SER
            // TODO: this is incorrect, what we need here is sum(len(KA))
            // only solution is to change the experiment interface where
            // estimate is calculated
            return get_symbolsperblock();
        case 2: // FER
            return 1;
        }
        // this should never happen
        return 0;
    }
    /*! \copydoc experiment::result_description()
     *
     * The description is a string which indicates symbol or
     * frame error rates or SKR.
     */
    std::string result_description(int i) const
    {
        assert(i >= 0 && i < count());
        switch (i) {
        case 0:
            return "SKR";
        case 1:
            return "SER";
        case 2:
            return "FER";
        }
        // this should never happen
        return "";
    }
    // @}
};

} // namespace libcomm

#endif
