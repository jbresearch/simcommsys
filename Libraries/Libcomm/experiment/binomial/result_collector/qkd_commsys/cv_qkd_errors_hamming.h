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
                       libbase::vector<bool> vector_s,
                       libbase::vector<bool>& key_KA,
                       libbase::vector<bool>& key_KB) const;
    /*! \copydoc experiment::count()
     * We count the number of symbol, frame errors and secret key rate for
     * CV-QKD.
     */
    int count() const
    {
        return 4;
    } // Accounts for the current results in updateresults().
    /*! \copydoc experiment::get_multiplicity()
     *
     * Since results are organized as (symbol,frame) error count, the
     * multiplicity is respectively the number of symbols and the number of
     * frames (=1) per sample.
     */
    int get_multiplicity(int i) const
    {
        assert(i >= 0 && i < count());
        return (i == 0) ? get_symbolsperblock() : 1;
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
            return "Sum of lengths of Key KA";
        case 1:
            // Sum of length of source which is the sequence of generated
            // coherent states. This is equal to N because it is fixed. In
            // qkd_commsys_simulator.cpp it is called the framesize.
            return "Sum of length of source";
        case 2:
            return "SER";
        case 3:
            return "FER";
        }
        return std::string();
    }
    // @}
};

} // namespace libcomm

#endif
