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

#ifndef __direct_block_blind_embedder_h
#define __direct_block_blind_embedder_h

#include "block_blind_embedder.h"
#include "blind_embedder.h"

namespace libcomm
{

/*!
 * \brief   Position-Independent Blockwise Blind Data Embedder/Extractor.
 * \author  Johann Briffa
 *
 * This class is a template definition for position-independent block blind
 * embedders; this needs to be specialized for actual use. Template parameter
 * defaults are provided here.
 */

template <class S,
          template <class> class C = libbase::vector,
          class dbl = double>
class direct_block_blind_embedder : public block_blind_embedder<S, C, dbl>
{
};

/*!
 * \brief   Position-Independent Vector Blind Data Embedder/Extractor
 * \author  Johann Briffa
 *
 * Vector implementation of a position-independent block blind embedder.
 */

template <class S, class dbl>
class direct_block_blind_embedder<S, libbase::vector, dbl>
    : public block_blind_embedder<S, libbase::vector, dbl>
{
public:
    /*! \name Type definitions */
    typedef libbase::vector<dbl> array1d_t;
    // @}
private:
    /*! \name User-defined parameters */
    std::shared_ptr<blind_embedder<S>>
        implementation; //! Implementation of embedding/extraction functions
                        // @}
protected:
    // Interface with derived classes
    void doembed(const int N,
                 const libbase::vector<int>& data,
                 const libbase::vector<S>& host,
                 libbase::vector<S>& tx);
    void doextract(const channel<S, libbase::vector>& chan,
                   const libbase::vector<S>& rx,
                   libbase::vector<array1d_t>& ptable);

public:
    // Informative functions
    int num_symbols() const { return implementation->num_symbols(); }

    // Description
    std::string description() const;

    // Serialization Support
    DECLARE_SERIALIZER(direct_block_blind_embedder)
};

/*!
 * \brief   Position-Independent Matrix Blind Data Embedder/Extractor
 * \author  Johann Briffa
 *
 * Matrix implementation of a position-independent block blind embedder.
 */

template <class S, class dbl>
class direct_block_blind_embedder<S, libbase::matrix, dbl>
    : public block_blind_embedder<S, libbase::matrix, dbl>
{
public:
    /*! \name Type definitions */
    typedef libbase::vector<dbl> array1d_t;
    // @}
private:
    /*! \name User-defined parameters */
    std::shared_ptr<blind_embedder<S>>
        implementation; //! Implementation of embedding/extraction functions
                        // @}
protected:
    // Interface with derived classes
    void doembed(const int N,
                 const libbase::matrix<int>& data,
                 const libbase::matrix<S>& host,
                 libbase::matrix<S>& tx);
    void doextract(const channel<S, libbase::matrix>& chan,
                   const libbase::matrix<S>& rx,
                   libbase::matrix<array1d_t>& ptable);

public:
    // Informative functions
    int num_symbols() const { return implementation->num_symbols(); }

    // Description
    std::string description() const;

    // Serialization Support
    DECLARE_SERIALIZER(direct_block_blind_embedder)
};

} // namespace libcomm

#endif
