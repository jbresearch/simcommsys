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

#ifndef __commsys_simulator_h
#define __commsys_simulator_h

#include "assertalways.h"
#include "commsys.h"
#include "config.h"
#include "experiment/experiment_binomial.h"
#include "experiment/results_collector.h"
#include "randgen.h"
#include "result_collector/commsys/fidelity_pos.h"
#include "serializer.h"
#include "source.h"
#include "vector.h"

#include <sstream>
#include <stdexcept>

namespace libcomm
{

/*!
 * \brief   Communication Systems Simulator Base.
 * \author  Johann Briffa
 *
 * Base class for simulator, to hold non-templated material that needs to be
 * referenced externally.
 */

class commsys_simulator_base : public experiment_binomial
{
public:
    // Interface for Results Collector
    typedef enum {
        SYMBOLS_PER_FRAME,
        SYMBOLS_PER_BLOCK,
        ALPHABET_SIZE
    } index_t;
};

/*!
 * \brief   Communication Systems Simulator.
 * \author  Johann Briffa
 *
 * \todo Clean up interface with commsys object, particularly in cycleonce()
 *
 * \todo Update interface to allow use of source<S> rather than source<int>
 */

template <class S>
class commsys_simulator : public commsys_simulator_base
{
public:
    /*! \name Type definitions */
    typedef float real;
    typedef libbase::vector<int> array1i_t;
    typedef libbase::vector<double> array1d_t;
    typedef libbase::vector<array1d_t> array1vd_t;
    // @}

protected:
    /*! \name Bound objects */
    std::shared_ptr<source<int>> src; //!< Source data sequence generator
    std::shared_ptr<commsys<S>> sys;  //!< Communication systems
    std::shared_ptr<results_collector<array1i_t>> rc; //!< Results collector
    // @}
    /*! \name Internal state */
    array1i_t last_event;
    // @}

    bool analyze_decode_iters = false;

protected:
    // Interface for Results Collector
    std::any get_value(const int index) const override
    {
        switch (index) {
        case SYMBOLS_PER_FRAME:
            return int(sys->getmodem()->input_block_size());
        case SYMBOLS_PER_BLOCK:
            return int(sys->input_block_size());
        case ALPHABET_SIZE:
            return int(sys->num_inputs());
        }
        // this should never happen
        throw std::out_of_range("Unknown parameter index " +
                                std::to_string(index));
    }

public:
    /*! \name Constructors / Destructors */
    /*!
     * \brief Copy constructor
     *
     * Initializes system with bound objects cloned from supplied system.
     */
    commsys_simulator(const commsys_simulator<S>& c)
        : src(std::dynamic_pointer_cast<source<int>>(c.src->clone())),
          sys(std::dynamic_pointer_cast<commsys<S>>(c.sys->clone())),
          rc(std::dynamic_pointer_cast<results_collector<array1i_t>>(
              c.rc->clone()))
    {
    }
    commsys_simulator() {}
    virtual ~commsys_simulator() {}
    // @}

    // Experiment parameter handling
    void seedfrom(libbase::random& r)
    {
        src->seedfrom(r);
        sys->seedfrom(r);
    }
    void set_parameters(const libbase::vector<double>& params) override
    {
        assertalways(params.size() == 1);
        sys->gettxchan()->set_parameter(params(0));
        sys->getrxchan()->set_parameter(params(0));
    }
    libbase::vector<double> get_parameters() const override
    {
        const double p = sys->gettxchan()->get_parameter();
        assert(p == sys->getrxchan()->get_parameter());

        libbase::vector<double> params;
        params.init(1);
        params(0) = p;
        return params;
    }
    int get_num_params() const override
    {
        return sys->gettxchan()->get_num_params();
    }

    // Experiment handling
    void sample(libbase::vector<double>& sample_result,
                libbase::vector<uint64_t>& sample_count) override;
    int result_count() const override
    {
        const fidelity_pos* rc_fidelity =
            dynamic_cast<const fidelity_pos*>(rc.get());
        if (analyze_decode_iters && !rc_fidelity)
            return rc->result_count() * sys->num_iter();
        else
            return rc->result_count();
    }
    std::string result_description(int i) const override
    {
        assert(i >= 0 && i < result_count());
        const int iter = i / rc->result_count();
        const int index = i % rc->result_count();
        std::ostringstream sout;
        sout << rc->result_description(index) << "_" << iter;
        return sout.str();
    }
    array1i_t get_event() const override { return last_event; }

    /*! \name Component object handles */
    //! Get communication system
    const std::shared_ptr<commsys<S>> getsystem() const { return sys; }
    //! Clear list of timers
    void reset_timers() { sys->reset_timers(); }
    //! Get the list of timings taken
    std::vector<double> get_timings() const { return sys->get_timings(); }
    //! Get the list of friendly names for timings taken
    std::vector<std::string> get_names() const { return sys->get_names(); }
    // @}

    // Description
    std::string description() const override;

    // Serialization Support
    DECLARE_SERIALIZER(commsys_simulator)
};

} // namespace libcomm

#endif
