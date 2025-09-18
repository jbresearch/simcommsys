/*!
 * \file
 *
 * Copyright (c) 2025 Mark Mizzi
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

#include "experiment/experiment_binomial.h"
#include "qkd_commsys.h"
#include "randgen.h"
#include "source.h"
#include "vector.h"
#include <sstream>

namespace libcomm
{

/*!
 * \brief   QKD Communication Systems Simulator.
 * \author  Mark Mizzi
 */

template <class S, class T, class R>
class qkd_commsys_simulator : public experiment_binomial, public R
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
    libbase::vector<bool> vector_s;
    // @}
    /*! \name Internal state */
    array1i_t last_event;

    libbase::random* rng_ =
        nullptr; // non-owning: set in seedfrom(), reused in sample()
    // @}

    // Class created to generate Bob's vector s of bool type.
    class bit_vector_generator
    {
    public:
        libbase::vector<bool> generate_vector(int k, libbase::random& r)
        {
            libbase::vector<bool> s(k);
            for (int i = 0; i < k; ++i) {
                s(i) = (r.ival(2) != 0);
            }
            return s;
        }
    } sgen;

public:
    /*! \name Constructors / Destructors */
    /*!
     * \brief Copy constructor
     *
     * Initializes system with bound objects cloned from supplied system.
     */
    qkd_commsys_simulator(const qkd_commsys_simulator<S, T, R>& c)
        : experiment_binomial(c), R(c), rng_(c.rng_), sgen(c.sgen)
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
    void sample(array1d_t& result) override;
    int count() const { return R::count(); }
    int get_multiplicity(int i) const
    {
        assert(i >= 0 && i < count());
        const int index = i % R::count();
        return R::get_multiplicity(index);
    }
    std::string result_description(int i) const
    {
        assert(i >= 0 && i < count());
        const int iter = i / R::count();
        const int index = i % R::count();
        std::ostringstream sout;
        sout << R::result_description(index) << "_" << iter;
        return sout.str();
    }
    array1i_t get_event() const { return last_event; }

    //  Getter to get the input bits from qkd_commsys.h which gets the inputs
    //  bits from the codec of the cvqkd_protocol.h.
    int get_codec_input_bits_k() { return sys->get_codec_input_bits_k(); }

    /*! \name Component object handles */
    //! Clear list of timers
    void reset_timers() { sys->reset_timers(); }
    //! Get the list of timings taken
    std::vector<double> get_timings() const { return sys->get_timings(); }
    //! Get the list of friendly names for timings taken
    std::vector<std::string> get_names() const { return sys->get_names(); }
    // @}

    // Required as they need to override the virtual methods found in
    // cv_qkd_errors_hamming.h.
    // TODO: Might have to remove these depending on the final results collector
    // I will implement.
    int get_symbolsperblock() const override
    {
        return sys ? sys->input_block_size()
                   : 0; // guard for default-constructed serializer path
    }

    int get_alphabetsize() const override { return 2; }

    // Description
    std::string description() const;

    // Serialization Support
    DECLARE_SERIALIZER(qkd_commsys_simulator)
};

} // namespace libcomm

#endif // __qkd_commsys_simulator
