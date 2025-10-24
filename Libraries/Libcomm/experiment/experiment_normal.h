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

#ifndef __experiment_normal_h
#define __experiment_normal_h

#include "experiment.h"

namespace libcomm
{

/*!
 * \brief   Experiment with normally distributed samples.
 * \author  Johann Briffa
 *
 * Implements the accumulator functions required by the experiment class,
 * moved from previous implementation in montecarlo.
 *
 * \todo Keep track of separate counts for each result, as in
 * experiment_binomial
 */

class experiment_normal : public experiment
{
    /*! \name Internal variables */
    libbase::vector<double> sum;   //!< Vector of result sums
    libbase::vector<double> sumsq; //!< Vector of result sum-of-squares
                                   // @}

protected:
    // Accumulator functions
    void derived_reset() override;
    void derived_accumulate_result(const libbase::vector<double>& sample_result) override;
    void derived_accumulate_state(const libbase::vector<double>& state) override;
    // @}

public:
    // Accumulator functions
    void get_state(libbase::vector<double>& state) const override;
    void estimate(libbase::vector<double>& estimate,
                  libbase::vector<double>& stderror) const override;
};

} // namespace libcomm

#endif
