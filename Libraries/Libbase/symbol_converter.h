/*!
 * \file
 *
 * Copyright (c) 2015 Johann A. Briffa
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

#ifndef SYMBOL_CONVERTER_H_
#define SYMBOL_CONVERTER_H_

#include "config.h"
#include "vector.h"

namespace libbase {

/*!
 * \brief   Symbols Aggregator
 * \author  Johann Briffa
 *
 * This class defines a symbol converter; this is a rate-1 mapping where
 * each symbol with a large alphabet encodes more than one symbol with a
 * small alphabet. Conversion operations are provided in each direction
 * (small->large and large->small) for symbols and for probability tables.
 *
 * \note Each large symbol must be representable by an integral number
 * of small symbols; additionally, when converting a sequence, the input
 * sequence must be representable by an integral number of output symbols.
 *
 * \tparam dbl Floating-point type for probability tables
 * \tparam dbl2 Floating-point type for internal computation (pre-normalization)
 */

template <class dbl, class dbl2>
class symbol_converter {
public:
   /*! \name Type definitions */
   typedef vector<dbl> array1d_t;
   typedef vector<int> array1i_t;
   typedef vector<array1d_t> array1vd_t;
   // @}

private:
   /*! \name Internal object representation */
   int S; //!< Alphabet size for small symbols
   int L; //!< Alphabet size for large symbols
   int k; //!< Number of small symbols per large symbol
   // @}

public:
   /*! \name Constructors & destructor */
   explicit symbol_converter(const int S, const int L) :
         S(S), L(L)
      {
      k = get_rate(S, L);
      }
   virtual ~symbol_converter()
      {
      }
   // @}

   /*! \name Helper functions */
   /*!
    * \brief Determines the number of small symbols per large symbol
    * \param[in]  S     Alphabet size for small symbols
    * \param[in]  L     Alphabet size for large symbols
    */
   static int get_rate(const int S, const int L)
      {
      const int k = int(round(log(double(L)) / log(double(S))));
      assertalways(k >= 1);
      assertalways(L == pow(S,k));
      return k;
      }
   // @}

   /*! \name Main interface */
   void aggregate_symbols(const array1i_t& in, array1i_t& out) const;
   void aggregate_probabilities(const array1vd_t& pin, array1vd_t& pout) const;
   void divide_symbols(const array1i_t& in, array1i_t& out) const;
   void divide_probabilities(const array1vd_t& pin, array1vd_t& pout) const;
   // @}
};

} /* namespace libbase */

#endif /* SYMBOL_CONVERTER_H_ */
