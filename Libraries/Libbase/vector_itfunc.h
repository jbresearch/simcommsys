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

#ifndef VECTOR_ITFUNC_H_
#define VECTOR_ITFUNC_H_

#include "vector.h"
#include "matrix.h"
#include "vectorutils.h"

/*!
 * \file
 * \brief   Information-Theory Utility Functions for Vectors.
 * \author  Johann Briffa
 *
 */

namespace libbase {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Keep track of problematic extrinsic computations
// 3 - Show results of extrinsic computation and results normalization
// NOTE: since this is a header, it may be included in other classes as well;
//       to avoid problems, the debug level is reset at the end of this file.
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 2
#endif

/*! \brief Compute extrinsic information
 *
 * \param[out] re extrinsic information
 * \param[in] ro 'full' posterior information
 * \param[in] ri (extrinsic) prior information
 *
 * Computes extrinsic information as re = ro/ri, except cases where ri=0, where
 * re=ro.
 *
 * \note re may point to the same memory as ro/ri, so care must be taken.
 */
template <class dbl>
void compute_extrinsic(vector<vector<dbl> >& re, const vector<vector<dbl> >& ro,
      const vector<vector<dbl> >& ri)
   {
#if DEBUG>=3
   std::cerr << "DEBUG (compute_extrinsic): ro = " << ro << std::endl;
   std::cerr << "DEBUG (compute_extrinsic): ri = " << ri << std::endl;
#endif
   // Handle the case where the prior information is empty
   if (ri.size() == 0)
      {
      re = ro;
      return;
      }
   // Determine size
   const int tau = ro.size();
   const int N = ro(0).size();
   // Check for validity
   assert(ri.size() == tau);
   assert(ri(0).size() == N);
   // Allocate space for re (if necessary)
   libbase::allocate(re, tau, N);
#if DEBUG>=2
   int count = 0;
#endif
   // Perform computation
   for (int i = 0; i < tau; i++)
      for (int x = 0; x < N; x++)
         if (ri(i)(x) > 0)
            {
            // normal extrinsic computation
            re(i)(x) = ro(i)(x) / ri(i)(x);
            }
         else if (ro(i)(x) == 0)
            {
            // if prior was also zero
            re(i)(x) = 0;
            }
         else
            {
            // problematic cases
#if DEBUG>=2
            count++;
#endif
            re(i)(x) = 0;
            }
#if DEBUG>=2
   if (count > 0)
      std::cerr << "DEBUG (compute_extrinsic): " << count
            << " problematic cases of " << tau * N << std::endl;
#endif
#if DEBUG>=3
   std::cerr << "DEBUG (compute_extrinsic): re = " << re << std::endl;
#endif
   }

/*!
 * \brief Normalize probability table
 *
 * \param[in] in Input vector of probabilities
 * \param[out] out Output (normalized) vector of probabilities
 *
 * The input probability table is normalized such that the sum of probabilities
 * is equal to 1, and copied to the output table.
 *
 * \note The input and output representation may be different, in which case
 * the values are automatically converted
 *
 * \note The output and input vectors may point to the same memory
 */
template <class real, class dbl>
void normalize(const vector<real>& in, vector<dbl>& out)
   {
   const int N = in.size();
   assert(N > 0);
   // check for numerical underflow
   real scale = in.sum();
   assertalways(scale > real(0));
   scale = real(1) / scale;
   // allocate result space
   out.init(N);
   // normalize and copy results
   for (int i = 0; i < N; i++)
      out(i) = dbl(in(i) * scale);
   }

/*!
 * \brief Normalize probability table
 *
 * \param[in] in Input vector of probabilities
 * \param[out] out Output (normalized) vector of probabilities
 *
 * The input probability table is normalized such that the sum of probabilities
 * at each index is equal to 1, and copied to the output table.
 *
 * \note The input and output representation may be different, in which case
 * the values are automatically converted
 *
 * \note The output and input vectors may point to the same memory
 */
template <class real, class dbl>
void normalize_results(const vector<vector<real> >& in,
      vector<vector<dbl> >& out)
   {
#if DEBUG>=3
   std::cerr << "DEBUG (normalize_results): in = " << in << std::endl;
#endif
   const int N = in.size();
   assert(N > 0);
   const int q = in(0).size();
   // allocate result space
   libbase::allocate(out, N, q);
   // normalize and copy results
   for (int i = 0; i < N; i++)
      normalize(in(i), out(i));
#if DEBUG>=3
   std::cerr << "DEBUG (normalize_results): out = " << out << std::endl;
#endif
   }

/*!
 * \brief Determine the entropy of the given pdf
 */
template <class real>
double compute_entropy(const vector<real>& p)
   {
   double H = 0;
   for (int i = 0; i < p.size(); i++)
      H += -p(i) * log2(p(i));
   return H;
   }

/*!
 * \brief Determine the average entropy of the given set of pdfs
 */
template <class real>
double compute_entropy(const vector<vector<real> >& ptable)
   {
   // determine sizes
   const int N = ptable.size();
   assert(N > 0);
   // compute entropy
   double H = 0;
   for (int i = 0; i < N; i++)
      H += compute_entropy(ptable(i));
   H /= N;
   return H;
   }

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG
#endif

} // end namespace

#endif /* VECTOR_ITFUNC_H_ */
