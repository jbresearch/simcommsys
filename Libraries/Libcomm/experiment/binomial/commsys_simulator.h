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

#include "commsys.h"
#include "config.h"
#include "experiment/experiment_binomial.h"
#include "randgen.h"
#include "serializer.h"
#include "source.h"
#include <sstream>

namespace libcomm
{

/*!
 * \brief   Communication Systems Simulator.
 * \author  Johann Briffa
 *
 * \todo Clean up interface with commsys object, particularly in cycleonce()
 *
 * \todo Update interface to allow use of source<S> rather than source<int>
 */

template <class S, class R>
class commsys_simulator : public experiment_binomial, public R
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
    // @}
    /*! \name Internal state */
    array1i_t last_event;
    // @}

protected:
    // System Interface for Results
    int get_symbolsperframe() const
    {
        return sys->getmodem()->input_block_size();
    }
    int get_symbolsperblock() const { return sys->input_block_size(); }
    int get_alphabetsize() const { return sys->num_inputs(); }

public:
    /*! \name Constructors / Destructors */
    /*!
     * \brief Copy constructor
     *
     * Initializes system with bound objects cloned from supplied system.
     */
    commsys_simulator(const commsys_simulator<S, R>& c)
        : src(std::dynamic_pointer_cast<source<int>>(c.src->clone())),
          sys(std::dynamic_pointer_cast<commsys<S>>(c.sys->clone()))
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
    void set_parameter(const double x)
    {
        sys->gettxchan()->set_parameter(x);
        sys->getrxchan()->set_parameter(x);
    }
    double get_parameter() const
    {
        const double p = sys->gettxchan()->get_parameter();
        assert(p == sys->getrxchan()->get_parameter());
        return p;
    }

    // Experiment handling
    void sample(array1d_t& result);
    int count() const { return R::count() * sys->num_iter(); }
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
    std::string description() const;

    // Serialization Support
    DECLARE_SERIALIZER(commsys_simulator)
};

} // namespace libcomm

#endif
