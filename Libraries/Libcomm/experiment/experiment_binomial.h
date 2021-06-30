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

#ifndef __experiment_binomial_h
#define __experiment_binomial_h

#include "experiment.h"

namespace libcomm
{

/*!
 * \brief   Experiment for estimation of a binomial proportion.
 * \author  Johann Briffa
 *
 * Implements the accumulator functions required by the experiment class.
 */

class experiment_binomial : public experiment
{
    /*! \name Internal variables */
    libbase::vector<double> sum; //!< Vector of result sums
                                 // @}

protected:
    // Accumulator functions
    void derived_reset();
    void derived_accumulate(const libbase::vector<double>& result);
    void accumulate_state(const libbase::vector<double>& state);
    // @}

public:
    // Accumulator functions
    void get_state(libbase::vector<double>& state) const;
    void estimate(libbase::vector<double>& estimate,
                  libbase::vector<double>& stderror) const;
};

} // namespace libcomm

#endif
