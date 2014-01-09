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

#ifndef __qim_h
#define __qim_h

#include "config.h"
#include "embedder.h"

namespace libcomm {

/*!
 * \brief   Quantization Index Modulation (QIM) Embedder/Extractor.
 * \author  Johann Briffa
 */

template <class S>
class qim : public embedder<S> {
private:
   /*! \name User-defined parameters */
   int M; //! Alphabet size in symbols
   S delta; //! QIM bin width
   double alpha; //! QIM distortion-compensation factor (0 < alpha <= 1)
   // @}
protected:
   //! Verifies that object is in a valid state
   void test_invariant() const
      {
      assert(M >= 2);
      assert(delta > 0);
      assert(alpha > 0.0 && alpha <= 1.0);
      }
   //! Embedder without distortion compensation
   const S Q(const int i, const double s) const
      {
      return S((round((s / delta - i) / M) * M + i) * delta);
      }
public:
   qim(const int M = 2, const S delta = 1, const double alpha = 1) :
      M(M), delta(delta), alpha(alpha)
      {
      }

   // Atomic embedder operations
   const S embed(const int i, const S s) const
      {
      return S(Q(i, s * alpha) + (1 - alpha) * s);
      }
   const int extract(const S& rx) const
      {
      // Find the symbol with the smallest discrepancy
      int d = 0;
      S best = std::abs(rx - embed(d, rx));
      for (int i = 1; i < M; i++)
         {
         const S diff = std::abs(rx - embed(i, rx));
         if (diff < best)
            {
            best = diff;
            d = i;
            }
         }
      return d;
      }

   // Informative functions
   int num_symbols() const
      {
      return M;
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER( qim)
};

} // end namespace

#endif
