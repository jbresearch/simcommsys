/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "ssis.h"
#include "fastsecant.h"
#include "itfunc.h"
#include "filter/atmfilter.h"
#include "filter/awfilter.h"
#include <cstdlib>
#include <sstream>
#include <vectorutils.h>

using libbase::fastsecant;
using libbase::cerf;

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
   static libbase::fastsecant erfinv(libbase::cerf);
   static bool initialized = false;
   if (!initialized)
      {
      erfinv.init(-0.99, 0.99, 1000);
      initialized = true;
      }
   const dbl gtilde = erfinv(2 * v - 1) * sqrt(2.0);
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
   sout << 1; // version number
   sout << A;
   sout << preprocess;
   return sout;
   }

/*!
 * \version 1 Initial version
 */

template <class S, class dbl>
std::istream& ssis<S, matrix, dbl>::serialize(std::istream& sin)
   {
   int version;
   sin >> libbase::eatcomments >> version;
   sin >> libbase::eatcomments >> A;
   int temp;
   sin >> libbase::eatcomments >> temp;
   assertalways(temp >=0 && temp < PP_UNDEFINED);
   preprocess = static_cast<pp_enum> (temp);
   return sin;
   }

// Explicit Realizations

// Matrix

template class ssis<int, matrix, double> ;
template <>
const serializer ssis<int, matrix, double>::shelper("blockembedder",
      "ssis<int,matrix>", ssis<int, matrix, double>::create);

template class ssis<float, matrix, double> ;
template <>
const serializer ssis<float, matrix, double>::shelper("blockembedder",
      "ssis<float,matrix>", ssis<float, matrix, double>::create);

template class ssis<double, matrix, double> ;
template <>
const serializer ssis<double, matrix, double>::shelper("blockembedder",
      "ssis<double,matrix>", ssis<double, matrix, double>::create);

} // end namespace
