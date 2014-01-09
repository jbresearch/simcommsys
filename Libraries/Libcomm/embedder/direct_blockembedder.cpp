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

#include "direct_blockembedder.h"
#include <cstdlib>
#include <sstream>
#include <vectorutils.h>

namespace libcomm {

using libbase::serializer;
using libbase::vector;
using libbase::matrix;

// *** Vector position-independent blockembedder ***

// Block modem operations

template <class S, class dbl>
void direct_blockembedder<S, vector, dbl>::doembed(const int N, const vector<
      int>& data, const vector<S>& host, vector<S>& tx)
   {
   // Check validity
   assertalways(data.size() == this->input_block_size());
   assertalways(host.size() == this->input_block_size());
   assertalways(N == this->num_symbols());
   // Inherit sizes
   const int tau = this->input_block_size();
   // Initialize results vector
   tx.init(tau);
   // Modulate encoded stream
   for (int i = 0; i < tau; i++)
      tx(i) = implementation->embed(data(i), host(i));
   }

template <class S, class dbl>
void direct_blockembedder<S, vector, dbl>::doextract(
      const channel<S, vector>& chan, const vector<S>& rx,
      vector<array1d_t>& ptable)
   {
   // Check validity
   assertalways(rx.size() == this->input_block_size());
   // Inherit sizes
   const int tau = this->input_block_size();
   const int M = this->num_symbols();
   // Allocate space for temporary results
   vector<vector<double> > ptable_double;
      {
      // Create a set of all possible transmitted symbols, at each timestep
      vector<vector<S> > tx;
      libbase::allocate(tx, tau, M);
      for (int t = 0; t < tau; t++)
         for (int x = 0; x < M; x++)
            tx(t)(x) = implementation->embed(x, rx(t));
      // Work out the probabilities of each possible signal
      chan.receive(tx, rx, ptable_double);
      }
   // Convert result
   ptable = ptable_double;
   }

// Description

template <class S, class dbl>
std::string direct_blockembedder<S, vector, dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Vector " << implementation->description();
   return sout.str();
   }

// Serialization Support

template <class S, class dbl>
std::ostream& direct_blockembedder<S, vector, dbl>::serialize(
      std::ostream& sout) const
   {
   sout << implementation;
   return sout;
   }

template <class S, class dbl>
std::istream& direct_blockembedder<S, vector, dbl>::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> implementation >> libbase::verify;
   return sin;
   }

// *** Matrix position-independent blockembedder ***

// Block modem operations

template <class S, class dbl>
void direct_blockembedder<S, matrix, dbl>::doembed(const int N, const matrix<
      int>& data, const matrix<S>& host, matrix<S>& tx)
   {
   // Check validity
   assertalways(data.size() == this->input_block_size());
   assertalways(host.size() == this->input_block_size());
   assertalways(N == this->num_symbols());
   // Inherit sizes
   const int rows = this->input_block_size().rows();
   const int cols = this->input_block_size().cols();
   // Initialize results matrix
   tx.init(this->input_block_size());
   // Modulate encoded stream
   for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++)
         tx(i, j) = implementation->embed(data(i, j), host(i, j));
   }

template <class S, class dbl>
void direct_blockembedder<S, matrix, dbl>::doextract(
      const channel<S, matrix>& chan, const matrix<S>& rx,
      matrix<array1d_t>& ptable)
   {
   // Check validity
   assertalways(rx.size() == this->input_block_size());
   // Inherit sizes
   const int rows = this->input_block_size().rows();
   const int cols = this->input_block_size().cols();
   const int M = this->num_symbols();
   // Allocate space for temporary results
   matrix<vector<double> > ptable_double;
      {
      // Create a set of all possible transmitted symbols, at each timestep
      matrix<vector<S> > tx;
      libbase::allocate(tx, rows, cols, M);
      for (int i = 0; i < rows; i++)
         for (int j = 0; j < cols; j++)
            for (int x = 0; x < M; x++)
               tx(i, j)(x) = implementation->embed(x, rx(i, j));
      // Work out the probabilities of each possible signal
      chan.receive(tx, rx, ptable_double);
      }
   // Convert result
   ptable = ptable_double;
   }

// Description

template <class S, class dbl>
std::string direct_blockembedder<S, matrix, dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Matrix " << implementation->description();
   return sout.str();
   }

// Serialization Support

template <class S, class dbl>
std::ostream& direct_blockembedder<S, matrix, dbl>::serialize(
      std::ostream& sout) const
   {
   sout << implementation;
   return sout;
   }

template <class S, class dbl>
std::istream& direct_blockembedder<S, matrix, dbl>::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> implementation >> libbase::verify;
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
   (vector)(matrix)
#define REAL_TYPE_SEQ \
   (double)

/* Serialization string: direct_blockembedder<type,container,real>
 * where:
 *      type = int | float | double
 *      container = vector | matrix
 *      real = double
 *              [real is the interface arithmetic type]
 */
#define INSTANTIATE(r, args) \
      template class direct_blockembedder<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer direct_blockembedder<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "blockembedder", \
            "direct_blockembedder<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(2,args)) ">", \
            direct_blockembedder<BOOST_PP_SEQ_ENUM(args)>::create);

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (SYMBOL_TYPE_SEQ)(CONTAINER_TYPE_SEQ)(REAL_TYPE_SEQ))

} // end namespace
