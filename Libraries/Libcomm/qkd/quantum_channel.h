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

#ifndef __quantum_channel_h
#define __quantum_channel_h

#include "assertalways.h"
#include "parametric.h"

namespace libcomm
{

class position_observable;
class momentum_observable;

class spin_computational;
class spin_hadamard;

/*!
 * \brief   Common Base for Quantum channel.
 * \author  Mark Mizzi
 *
 * We use the Heisenberg picture, so that quantum channels act on observables.
 * We use the visitor pattern to model these actions, so the base interface
 * contains methods (called \ref transmit()) for "visiting" each concrete
 * observable type. By default these methods fail, but a concrete quantum
 * channel subclass is meant to override the methods for observables that they
 * support.
 */
class quantum_channel : public parametric
{
public:
    //! \name Visitor interface methods for observables
    virtual void transmit(position_observable&) const
    {
        failwith("Not implemented.");
    }
    virtual void transmit(momentum_observable&) const
    {
        failwith("Not implemented.");
    }
    virtual void transmit(spin_computational&) const
    {
        failwith("Not implemented.");
    }
    virtual void transmit(spin_hadamard&) const
    {
        failwith("Not implemented.");
    }
    //! @}

    virtual ~quantum_channel() {}
};

} // end namespace libcomm

#endif // __quantum_channel_h