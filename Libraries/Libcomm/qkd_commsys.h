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

#ifndef __qkd_commsys_h
#define __qkd_commsys_h

#include "commsys.h"

#include "fsm.h"
#include "itfunc.h"
#include "mapper/map_straight.h"
#include "qkd/observable.h"
#include "qkd/qkd_postprocessor.h"
#include "qkd/quantum_channel.h"
#include "qkd/quantum_state.h"
#include "secant.h"
#include "timer.h"
#include "vector.h"

#include <iostream>
#include <memory>
#include <sstream>

namespace libcomm
{

/*!
 * \brief   Common Base for QKD System.
 * \author  Mark Mizzi
 */

template <class S, class T>
class basic_qkd_commsys : public instrumented
{
public:
    /*! \name Type definitions */
    typedef libbase::vector<double> array1d_t;
    // @}

protected:
    /*! \name Bound objects */
    std::unique_ptr<quantum_channel> bob_channel;
    std::unique_ptr<quantum_channel> alice_channel;
    std::unique_ptr<qkd_postprocessor> protocol;
    // @}
#ifndef NDEBUG
    bool lastframecorrect;
    C<int> lastsource;
#endif
protected:
    /*! \name Setup functions */
    void init();
    void free();
    // @}
public:
    basic_commsys() {}
    virtual ~basic_qkd_commsys() { free(); }
    // @}

    /*! \name Communication System Setup */
    virtual void seedfrom(libbase::random& r);
    //! Get error-control codec
    std::shared_ptr<codec<C>> getcodec() const { return cdc; }
    //! Get symbol mapper
    std::shared_ptr<mapper<C>> getmapper() const { return map; }
    //! Get modulation scheme
    std::shared_ptr<blockmodem<S, C>> getmodem() const { return mdm; }
    //! Get channel model - transmitter side
    std::shared_ptr<channel<S, C>> gettxchan() const { return txchan; }
    //! Get channel model - receiver side
    std::shared_ptr<channel<S, C>> getrxchan() const { return rxchan; }
    // @}

    /*! \name Communication System Interface */
    //! Perform complete encode path
    virtual C<S> encode_path(const C<int>& source);
    //! Perform channel transmission
    virtual C<S> transmit(const C<S>& transmitted);
    //! Perform complete receive path, except for final decoding
    virtual void receive_path(const C<S>& received);
    //! Perform after-demodulation receive path, except for final decoding
    virtual void softreceive_path(const C<array1d_t>& ptable_mapped);
    //! Perform all decoding iterations, with hard decision
    virtual void decode(C<int>& decoded);
    //! Perform all decoding iterations, with hard decision; also returning the
    //! codeword at each iteration in the process.
    virtual void decode(libbase::vector<C<int>>& decoded);
    // @}

    /*! \name Informative functions */
    //! Number of iterations to perform
    virtual int num_iter() const { return cdc->num_iter(); }
    //! Overall mapper rate
    double rate() const { return cdc->rate() * map->rate(); }
    //! Input alphabet size (number of valid symbols)
    int num_inputs() const { return cdc->num_inputs(); }
    //! Output alphabet size (number of valid symbols)
    int num_outputs() const { return mdm->num_symbols(); }
    //! Input (ie. source/decoded) block size in symbols
    libbase::size_type<C> input_block_size() const
    {
        return cdc->input_block_size();
    }
    //! Output (ie. transmitted/received) block size in symbols
    libbase::size_type<C> output_block_size() const
    {
        return mdm->output_block_size();
    }
    // @}

    //! Clear list of timers
    void reset_timers()
    {
        // clear list of timers we're keeping
        instrumented::reset_timers();
        // clear list of timers for all components
        protocol->reset_timers();
    }

    // Description
    virtual std::string description() const;
    std::ostream& serialize(std::ostream& sout) const;
    std::istream& serialize(std::istream& sin);
};

} // namespace libcomm

#endif // __qkd_commsys_h