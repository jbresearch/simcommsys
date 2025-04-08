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

#ifndef __parametric_h
#define __parametric_h

#include "config.h"
#include "vector.h"

namespace libcomm
{

/*!
 * \brief   Parametric Class Interface.
 * \author  Mark Mizzi
 *
 * Defines a class that takes multiple parameters.
 */

class parametric
{
public:
    /*! \name Constructors / Destructors */
    virtual ~parametric() {}
    // @}

    /*! \name Parameter handling */
    //! Set the characteristic parameters
    virtual void set_parameters(const libbase::vector<double>& x) = 0;
    //! Get the characteristic parameters
    virtual libbase::vector<double> get_parameters() const = 0;
    virtual int get_num_params() const = 0;
    // @}
};

/*!
 * \brief   Mono parametric Class Interface.
 * \author  Johann Briffa
 *
 * Defines a class that takes a single scalar parameter.
 */

class mono_parametric : public parametric
{
public:
    /*! \name Constructors / Destructors */
    virtual ~mono_parametric() {}
    // @}

    void set_parameters(const libbase::vector<double>& x) override
    {
        assertalways(x.size() == 1);
        this->set_parameter(x(0));
    }
    libbase::vector<double> get_parameters() const override
    {
        libbase::vector<double> params;
        params.init(1);
        params(0) = this->get_parameter();
        return params;
    }
    int get_num_params() const override { return 1; }

    /*! \name Parameter handling */
    //! Set the characteristic parameter
    virtual void set_parameter(const double x) = 0;
    //! Get the characteristic parameter
    virtual double get_parameter() const = 0;
    // @}
};

} // namespace libcomm

#endif
