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

#ifndef __commsys_timer_h
#define __commsys_timer_h

#include "config.h"
#include "experiment/binomial/commsys_simulator.h"
#include "experiment/experiment_normal.h"
#include "experiment/results_collector.h"
#include "vector.h"

#include <string>

namespace libcomm
{

/*!
 * \brief   Communication System Simulator - Timing collector.
 * \author  Johann Briffa
 *
 * A variation on the regular commsys_simulator object, returning component
 * timings as main result.
 */
template <class S>
class commsys_timer : public experiment_normal
{
private:
    commsys_simulator<S> simulator; //!< Base simulator object
    std::vector<double> timings;    //!< List of timings from last cycle
    std::vector<std::string> names; //!< List of timer names from last cycle

protected:
    // Interface for Results Collector
    std::any get_value(const int index) const override
    {
        switch (index) {
        }
        // this should never happen
        throw std::out_of_range("Unknown parameter index " +
                                std::to_string(index));
    }

public:
    // Experiment parameter handling
    void seedfrom(libbase::random& r) { simulator.seedfrom(r); }
    void set_parameters(const libbase::vector<double>& params) override
    {
        simulator.set_parameters(params);
    }
    libbase::vector<double> get_parameters() const override
    {
        return simulator.get_parameters();
    }

    // Experiment handling
    void sample(libbase::vector<double>& sample_result,
                libbase::vector<uint64_t>& sample_count) override
    {
        // Run the system simulation
        libbase::vector<double> temp_result;
        libbase::vector<uint64_t> temp_count;
        simulator.sample(temp_result, temp_count);
        // Collect timings
        timings = simulator.get_timings();
        names = simulator.get_names();
#ifdef DEBUG
        std::clog << "Timings: " << libbase::vector<std::string>(names);
#endif
        // Copy over timings as results
        sample_result = libbase::vector<double>(timings);
        // TODO: decide if we need to do anything with the count
    }
    int result_count() const override
    {
        const size_t N = timings.size();
        assert(N == names.size());
        assert(N > 0);
        return N;
    }
    std::string result_description(int i) const
    {
        assert(i >= 0 && i < int(names.size()));
        return names[i];
    }
    libbase::vector<int> get_event() const { return simulator.get_event(); }

    // Description
    std::string description() const
    {
        return "Timed " + simulator.description();
    }

    // Serialization Support
    DECLARE_SERIALIZER(commsys_timer)
};

} // namespace libcomm

#endif
