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

#ifndef __results_collector_h
#define __results_collector_h

#include "config.h"
#include "serializer.h"
#include "vector.h"
#include "queryable.h"
#include <string>

namespace libcomm
{

/*!
 * \brief   Results Collector Interface.
 * \author  Johann Briffa
 *
 * Templated interface for collection of results in experiments. Currently this
 * is only used in commsys_simulator and related objects. However, the interface
 * is intended to be sufficiently general for wider use.
 */
template <class T>
class results_collector : public libbase::serializable
{
public:
    virtual ~results_collector() {}
    /*! \name Public interface */
    /*!
     * \brief Initialise the results collector
     *
     * Initialise this results collector, getting the necessary values from the
     * supplied system.
     */
    virtual void init(const queryable& system) = 0;
    /*!
     * \brief Update accumulated results
     *
     * Compute the necessary statistics and update the supplied accumulated
     * results vector accordingly.
     */
    virtual void updateresults(libbase::vector<double>& result,
                               const T& source,
                               const T& decoded) const = 0;
    // \copydoc experiment::result_count()
    virtual int result_count() const = 0;
    // \copydoc experiment::result_multiplicity()
    virtual int result_multiplicity(int i) const = 0;
    // \copydoc experiment::result_description()
    virtual std::string result_description(int i) const = 0;
    // @}

    /*! \name Description */
    //! Human-readable experiment description
    virtual std::string description() const = 0;
    // @}

    // Serialization Support
    DECLARE_BASE_SERIALIZER(results_collector)
};

} // namespace libcomm

#endif
