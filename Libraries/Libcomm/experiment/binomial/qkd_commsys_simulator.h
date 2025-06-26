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
    std::unique_ptr<source<S>> src;         //!< Source data sequence generator
    std::unique_ptr<qkd_commsys<S, T>> sys; //!< Communication systems
    // @}
    /*! \name Internal state */
    array1i_t last_event;
    // @}

public:
    /*! \name Constructors / Destructors */
    /*!
     * \brief Copy constructor
     *
     * Initializes system with bound objects cloned from supplied system.
     */
    qkd_commsys_simulator(const qkd_commsys_simulator<S, T, R>& c)
         : src(std::dynamic_pointer_cast<source<S>>(c.src->clone())),
           sys(std::dynamic_pointer_cast<qkd_commsys<S, T>>(c.sys->clone()))
    {
    }
    qkd_commsys_simulator() {}
    virtual ~qkd_commsys_simulator() {}
    // @}

    // Experiment parameter handling
    void seedfrom(libbase::random& r)
    {
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

    /*! \name Component object handles */
    //! Clear list of timers
    void reset_timers() { sys->reset_timers(); }
    //! Get the list of timings taken
    std::vector<double> get_timings() const { return sys->get_timings(); }
    //! Get the list of friendly names for timings taken
    std::vector<std::string> get_names() const { return sys->get_names(); }
    // @}

    // Description
    std::string description() const;

    // Serialization Support
    DECLARE_SERIALIZER(qkd_commsys_simulator)
};

} // namespace libcomm

#endif // __qkd_commsys_simulator
