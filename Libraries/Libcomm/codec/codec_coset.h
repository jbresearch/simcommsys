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

#ifndef __codec_coset_h
#define __codec_coset_h

#include "codec.h"
#include "config.h"
#include "hard_decision.h"

namespace libcomm
{

/*!
 * \brief   Channel Codec with non-zero coset decoding.
 * \author  Johann Briffa
 */

template <template <class> class C = libbase::vector, class dbl = double>
class codec_coset : public codec<C, dbl>
{
private:
    // Shorthand for class hierarchy
    typedef codec<C, dbl> Base;

public:
    /*! \name Type definitions */
    typedef libbase::vector<dbl> array1d_t;
    // @}

protected:
    /*! \name Interface with derived classes */
    //! \copydoc init_decoder()
    virtual void do_init_decoder(const C<array1d_t>& ptable,
                                 const C<int>& syndrome) = 0;
    // @}
public:
    /*! \name Constructors / Destructors */
    virtual ~codec_coset() {}
    // @}

    /*! \name Codec operations */
    // Inherit receiver translation process from base class
    using Base::init_decoder;
    /*!
     * \brief Receiver translation process (with given priors)
     * \param[in] ptable Likelihoods of each possible encoded symbol at every
     * index 
     * \param[in] syndrome Syndrome defining the coset against which to decode
     *
     * This function initializes the decoder with the probability tables for
     * each encoded symbol as received from the blockmodem, and the syndrome
     * defining the coset on which we need to do the decoding.
     * This function should be called before the first decode iteration
     * for each block.
     */
    void init_decoder(const C<array1d_t>& ptable, const C<int>& syndrome)
    {
        libbase::cputimer t("t_init_decoder");
        this->advance_if_dirty();
        do_init_decoder(ptable, syndrome);
        this->mark_as_dirty();
        this->add_timer(t);
    }
    /*!
     * \brief Calculate the syndrome for a given codeword
     * \param[in] codeword The codeword for which we want to calculate the syndrome
     * \param[out] syndrome The returned syndrome
     */
    virtual void calculate_syndrome(const C<int>& codeword, C<int>& syndrome) = 0;
    // @}

    // Serialization Support
    // NOTE that this NEEDS to be here despite codec_coset being a subclass of
    // codec which also has an invocation of this template. This is because
    // there are places in the code where we serialize a
    // std::shared_ptr<codec_coset>, and this cannot be done using the methods
    // for serializing std::shared_ptr<codec> as C++ does not have covariant
    // typing rules.
    DECLARE_BASE_SERIALIZER(codec_coset)
};

} // namespace libcomm

#endif
