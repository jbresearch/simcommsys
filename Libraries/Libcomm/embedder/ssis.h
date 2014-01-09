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

#ifndef __ssis_h
#define __ssis_h

#include "blockembedder.h"
#include "embedder.h"

namespace libcomm {

/*!
 * \brief   Spread Spectrum Image Steganography Embedder/Extractor.
 * \author  Johann Briffa
 *
 * This class is a template definition for SSIS-based embedders;
 * this needs to be specialized for actual use. Template parameter
 * defaults are provided here.
 */

template <class S, template <class > class C = libbase::matrix,
      class dbl = double>
class ssis : public blockembedder<S, C, dbl> {
};

/*!
 * \brief   SSIS Matrix Embedder/Extractor
 * \author  Johann Briffa
 *
 * Matrix implementation of Marvel et al.'s SSIS algorithm for data embedding
 * in images.
 */

template <class S, class dbl>
class ssis<S, libbase::matrix, dbl> : public blockembedder<S, libbase::matrix,
      dbl> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   // @}
private:
   /*! \name User-defined parameters */
   double A; //!< Embedding strength (amplitude)
   enum pp_enum {
      PP_NONE, //!< No pre-processing
      PP_AW_EMBED, //!< Adaptive Wiener de-noising, use embedding strength
      PP_AW_MATLAB,//!< Adaptive Wiener de-noising, Matlab estimator
      PP_UNDEFINED
   } preprocess;
   // @}
   /*! \name Internal representation */
   mutable libbase::randgen r; //!< Uniform sequence generator
   mutable libbase::matrix<dbl> u; //!< Uniform sequence for current block
#ifndef NDEBUG
   mutable int frame; //!< Frame counter since seeding
#endif
   // @}
protected:
   /*! \name Internal helper operations */
   //! Piece-wise Linear Modulator
   static double plmod(const dbl u);
   /*!
    * \brief Embed a single symbol
    * \param   data Index into the symbol alphabet (data to embed)
    * \param   host Host value into which to embed data
    * \param   u    Value from uniform sequence corresponding to this position
    * \return  Stego-value, encoding the given data
    */
   static const S embed(const int data, const S host, const dbl u, const dbl A);
   /*!
    * \brief Extract a single symbol
    * \param   rx Received (possibly corrupted) stego-value
    * \return  Index corresponding to most-likely transmitted symbol
    */
   //const int extract(const S& rx) const;
   // @}
   // Interface with derived classes
   void advance() const;
   void doembed(const int N, const libbase::matrix<int>& data,
         const libbase::matrix<S>& host, libbase::matrix<S>& tx);
   void doextract(const channel<S, libbase::matrix>& chan,
         const libbase::matrix<S>& rx, libbase::matrix<array1d_t>& ptable);
public:
   // Setup functions
   void seedfrom(libbase::random& r)
      {
      libbase::int32u seed = r.ival();
#ifndef NDEBUG
      frame = 0;
      libbase::trace << "DEBUG (ssis): Seeding with " << seed << std::endl;
#endif
      this->r.seed(seed);
      advance();
      }

   // Informative functions
   int num_symbols() const
      {
      return 2;
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(ssis)
};

} // end namespace

#endif
