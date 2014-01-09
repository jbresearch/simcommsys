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

#include "ssis.h"
#include "filter/atmfilter.h"
#include "filter/awfilter.h"
#include <cstdlib>
#include <sstream>
#include <vectorutils.h>

#include <boost/math/special_functions/erf.hpp>

namespace libcomm {

using libbase::serializer;
using libbase::vector;
using libbase::matrix;

// *** Matrix SSIS ***

// Internal helper operations

template <class S, class dbl>
double ssis<S, matrix, dbl>::plmod(const dbl u)
   {
   if (u < 0.5)
      return u + 0.5;
   else if (u > 0.5)
      return u - 0.5;
   else
      return 0;
   }

template <class S, class dbl>
const S ssis<S, matrix, dbl>::embed(const int data, const S host, const dbl u,
      const dbl A)
   {
   // Modulate uniform sequence
   const dbl v = (data == 0) ? u : plmod(u);
   // Convert to Gaussian
   const dbl gtilde = boost::math::erf_inv(2 * v - 1) * sqrt(2.0);
   // Scale and embed
   return host + S(gtilde * A);
   }

// Block modem operations

template <class S, class dbl>
void ssis<S, matrix, dbl>::advance() const
   {
   // Inherit sizes
   const int rows = this->input_block_size().rows();
   const int cols = this->input_block_size().cols();
   // Generate uniform sequence for current block
   u.init(this->input_block_size());
   for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++)
         u(i, j) = r.fval_halfopen();
#ifndef NDEBUG
   frame++;
   libbase::trace << "DEBUG (ssis): Advanced to frame " << frame << std::endl;
#endif
   }

template <class S, class dbl>
void ssis<S, matrix, dbl>::doembed(const int N, const matrix<int>& data,
      const matrix<S>& host, matrix<S>& tx)
   {
   // Check validity
   assertalways(data.size() == this->input_block_size());
   assertalways(host.size() == this->input_block_size());
   assertalways(N == this->num_symbols());
   // Inherit sizes
   const int rows = this->input_block_size().rows();
   const int cols = this->input_block_size().cols();
   // Pre-processing
   matrix<S> pp_host(host.size());
   switch (preprocess)
      {
      case PP_NONE:
         pp_host = host;
         break;
      case PP_AW_EMBED:
         // Adaptive Wiener de-noising, use embedding strength
         {
         const int d = 1;
         const double noise = A * A;
         libimage::awfilter<S> filter(d, noise);
         filter.apply(host, pp_host);
         }
         break;
      case PP_AW_MATLAB:
         // Adaptive Wiener de-noising, Matlab estimator
         {
         const int d = 1;
         libimage::awfilter<S> filter(d);
         filter.apply(host, pp_host);
         }
         break;
      default:
         failwith("Unknown pre-processing method");
      }
   // Initialize results matrix
   tx.init(this->input_block_size());
   // Modulate encoded stream
   for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++)
         tx(i, j) = embed(data(i, j), pp_host(i, j), u(i, j), A);
   }

template <class S, class dbl>
void ssis<S, matrix, dbl>::doextract(const channel<S, matrix>& chan,
      const matrix<S>& rx, matrix<array1d_t>& ptable)
   {
   // Check validity
   assertalways(rx.size() == this->input_block_size());
   // Inherit sizes
   const int rows = this->input_block_size().rows();
   const int cols = this->input_block_size().cols();
   const int M = this->num_symbols();
   // Estimate embedded message with ATM filter
   matrix<S> est(rx.size());
   const int d = 1;
   const int alpha = 1;
   libimage::atmfilter<S> filter(d, alpha);
   filter.apply(rx, est);
   est = rx - est;
   // Allocate space for temporary results
   matrix<vector<double> > ptable_double;
      {
      // Create a set of all possible transmitted symbols, at each timestep
      matrix<vector<S> > tx;
      libbase::allocate(tx, rows, cols, M);
      for (int i = 0; i < rows; i++)
         for (int j = 0; j < cols; j++)
            for (int x = 0; x < M; x++)
               tx(i, j)(x) = embed(x, 0, u(i, j), A);
      // Work out the probabilities of each possible signal
      chan.receive(tx, est, ptable_double);
      }
   // Convert result
   ptable = ptable_double;
   }

// Description

template <class S, class dbl>
std::string ssis<S, matrix, dbl>::description() const
   {
   std::ostringstream sout;
   sout << "SSIS embedder (A=" << A;
   switch (preprocess)
      {
      case PP_NONE:
         break;
      case PP_AW_EMBED:
         sout << ", AW pre-denoiser [embedding strength]";
         break;
      case PP_AW_MATLAB:
         sout << ", AW pre-denoiser [Matlab estimator]";
         break;
      default:
         failwith("Unknown pre-processing method");
      }
   sout << ")";
   return sout.str();
   }

// Serialization Support

template <class S, class dbl>
std::ostream& ssis<S, matrix, dbl>::serialize(std::ostream& sout) const
   {
   sout << "# Version" << std::endl;
   sout << 1 << std::endl;
   sout << "# Embedding strength (amplitude)" << std::endl;
   sout << A << std::endl;
   sout << "# Pre-processing (0=none, 1=AW(ES), 2=AW(Matlab))" << std::endl;
   sout << preprocess << std::endl;
   return sout;
   }

/*!
 * \version 1 Initial version
 */

template <class S, class dbl>
std::istream& ssis<S, matrix, dbl>::serialize(std::istream& sin)
   {
   int version;
   sin >> libbase::eatcomments >> version >> libbase::verify;
   sin >> libbase::eatcomments >> A >> libbase::verify;
   int temp;
   sin >> libbase::eatcomments >> temp >> libbase::verify;
   assertalways(temp >=0 && temp < PP_UNDEFINED);
   preprocess = static_cast<pp_enum> (temp);
   return sin;
   }

} // end namespace

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;
using libbase::matrix;
using libbase::vector;

#define SYMBOL_TYPE_SEQ \
   (int)(float)(double)
#define CONTAINER_TYPE_SEQ \
   (matrix)
#define REAL_TYPE_SEQ \
   (double)

/* Serialization string: ssis<type,container,real>
 * where:
 *      type = int | float | double
 *      container = matrix
 *      real = double
 *              [real is the interface arithmetic type]
 */
#define INSTANTIATE(r, args) \
      template class ssis<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer ssis<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "blockembedder", \
            "ssis<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(2,args)) ">", \
            ssis<BOOST_PP_SEQ_ENUM(args)>::create);

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (SYMBOL_TYPE_SEQ)(CONTAINER_TYPE_SEQ)(REAL_TYPE_SEQ))

} // end namespace
