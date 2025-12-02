/*!
 * \file
 *
 * Copyright (c) 2025 Mark Mizzi, Aaron Abela
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

#ifndef __qkd_commsys_simulator_h
#define __qkd_commsys_simulator_h

#include "assertalways.h"
#include "experiment/experiment_binomial.h"
#include "experiment/results_collector.h"
#include "qkd_commsys.h"
#include "randgen.h"
#include "serializer.h"
#include "source.h"
#include "vector.h"
#include <sstream>
#include <stdexcept>

namespace libcomm
{

/*!
 * \brief   QKD Communication Systems Simulator.
 * \author  Mark Mizzi, Aaron Abela
 */

class qkd_commsys_simulator_base : public experiment_binomial
{
public:
    // Interface for Results Collector
    typedef enum {
        SOURCE_LENGTH,
        SECRET_KEY_LENGTH,
        ALPHABET_SIZE
    } index_t;
};

template <class S, class T>
class qkd_commsys_simulator : public qkd_commsys_simulator_base
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

    /* TODO: Discuss with Johann both the src and sys originally were
     * unique_ptrs but this was causing issues during compilation. Changed to
     * shared_ptrs.*/

    std::shared_ptr<source<S>> src;         //!< Source data sequence generator
    std::shared_ptr<qkd_commsys<S, T>> sys; //!< Communication systems
    std::shared_ptr<results_collector<libbase::vector<bool>>> rc; //!< Results collector
    // @}
    /*! \name Internal state */
    array1i_t last_event;

    libbase::random* rng_ =
        nullptr; // non-owning: set in seedfrom(), reused in sample()
    // @}

    // Interface for Results Collector
    std::any get_value(const int index) const override
    {
        switch (index) {
        case SOURCE_LENGTH:
            return int(sys->input_block_size()); // Equivalent to the framesize serialized in qkd_commsys.cpp
        case SECRET_KEY_LENGTH:
            return int(sys->get_protocol()->calculate_finite_size_effects_secret_key_length()); // Length of the final secret key.
        case ALPHABET_SIZE:
            return int(2);
        }
        // this should never happen
        throw std::out_of_range("Unknown parameter index " +
                                std::to_string(index));
    }

public:
    /*! \name Constructors / Destructors */

    /* Constructor */
    qkd_commsys_simulator(std::shared_ptr<libbase::random> rng_t,
                        std::shared_ptr<source<S>> src_gen_t,
                        std::shared_ptr<qkd_commsys<S, T, libbase::vector>> sys_t)
        : qkd_commsys_simulator_base()
        , src(src_gen_t)
        , sys(sys_t)
        , rng_(rng_t.get())
    {
        // Pass the pointer to the qkd_commsy_simulator object to the qkd_commsys system.
        sys->init(this);
    }

    /*!
     * \brief Copy constructor
     *
     * Initializes system with bound objects cloned from supplied system.
     */
    qkd_commsys_simulator(const qkd_commsys_simulator<S, T>& c)
        : qkd_commsys_simulator_base(c), rng_(c.rng_)
    {
        if (c.src)
            src = std::dynamic_pointer_cast<source<S>>(c.src->clone());
        if (c.sys)
            sys = std::dynamic_pointer_cast<qkd_commsys<S, T>>(c.sys->clone());
    }
    // @}

    qkd_commsys_simulator() {}

    virtual ~qkd_commsys_simulator() {}
    // @}

    // Experiment parameter handling
    void seedfrom(libbase::random& r) // do I need override to check?
    {
        rng_ = &r; // Stores RNG for later use.
        src->seedfrom(r);
        sys->seedfrom(r);
    }

    // Gets the source generator
    std::shared_ptr<source<S>> get_src_gen() { return src; }

    /*! \name Parametric interface */
    void set_parameters(const libbase::vector<double>& params) override
    {
        sys->set_parameters(params);
    }

    libbase::vector<double> get_parameters() const override
    {
        return sys->get_parameters();
    }

    int get_num_params() const override { return sys->get_num_params(); }
    // @}

    // Experiment handling
    void sample(libbase::vector<double>& sample_result,
                libbase::vector<uint64_t>& sample_count) override;

    int result_count() const override { return rc->result_count(); }

    std::string result_description(int i) const override
    {
        assert(i >= 0 && i < result_count());
        const int iter = i / rc->result_count();
        const int index = i % rc->result_count();
        std::ostringstream sout;
        sout << rc->result_description(index) << "_" << iter;
        return sout.str();
    }
    array1i_t get_event() const { return last_event; }
    
    /*! \name Component object handles */
     //! Get communication system
    const std::shared_ptr<qkd_commsys<S, T>> getsystem() const { return sys; }
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
    DECLARE_SERIALIZER(qkd_commsys_simulator)
};

} // namespace libcomm

#endif // __qkd_commsys_simulator
