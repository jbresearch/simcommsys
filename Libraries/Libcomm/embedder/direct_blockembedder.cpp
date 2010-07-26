/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
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
   sin >> libbase::eatcomments >> implementation;
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
   sin >> libbase::eatcomments >> implementation;
   return sin;
   }

// Explicit Realizations

// Vector

template class direct_blockembedder<int, vector, double> ;
template <>
const serializer direct_blockembedder<int, vector, double>::shelper(
      "blockembedder", "blockembedder<int,vector>", direct_blockembedder<int,
            vector, double>::create);

// Matrix

template class direct_blockembedder<int, matrix, double> ;
template <>
const serializer direct_blockembedder<int, matrix, double>::shelper(
      "blockembedder", "blockembedder<int,matrix>", direct_blockembedder<int,
            matrix, double>::create);

} // end namespace
