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

#ifndef __qkd_observable_h
#define __qkd_observable_h

#include "assertalways.h"
#include "qkd/quantum_channel.h"

namespace libcomm
{

class qubit;
class gaussian_state;

class epr_beam;
class entangled_qubit_pair;

/*!
 * \brief   Common Base for Observables.
 * \author  Mark Mizzi
 *
 * Observables are "visitors" of quantum state types.
 * This means that we need a "visiting" method in the base class for each
 * possible quantum state (called \ref measure()). By default these are not
 * implemented, concrete subclasses implement the specific methods for the
 * quantum states they support.entangled_qubit_pair The interface differs for
 * non-entangled and entangled quantum state types as for the latter we need to
 * specify which part of the state we are measuring. The \name T type parameter
 * refers to the output produced by the measurement, e.g. double for position,
 * momentum, bool for spin in some basis. Quantum channels act on observables as
 * we use the Heisenberg picture, so quantum channels are "visitors" of
 * observables.
 */
template <typename T>
class observable
{
public:
    //! \name Visitor interface methods for regular (non-entangled) states
    virtual T measure(qubit&) const { failwith("Not implemented.") }
    virtual T measure(gaussian_state&) const { failwith("Not implemented.") }
    //! @}

    //! \name Visitor interface methods for entangled states
    virtual T measure(entangled_qubit_pair&, int) const
    {
        failwith("Not implemented.")
    }
    virtual T measure(epr_beam&, int) const { failwith("Not implemented.") }
    //! @}

    /** \brief Implements the other side of the visitor pattern, which calls the
     * right transmit() method of \name quantum_channel.
     *
     * Implementation in subclasses should always be to call the \name
     * quantum_channel's transmit() method with *this.
     */
    virtual transmit(const quantum_channel&) = 0;

    virtual ~observable() {}
};

} // end namespace libcomm

#endif // __qkd_observable_h