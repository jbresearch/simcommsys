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

#ifndef __source_memoryless_h
#define __source_memoryless_h

#include "config.h"
#include "source.h"
#include "serializer.h"
#include <sstream>

namespace libcomm {

/*!
 * \brief   Memoryless source.
 * \author  Johann Briffa
 *
 * Implements a source that returns a random symbol from an alphabet with a
 * given distribution.
 */

template <class S, template <class > class C = libbase::vector>
class memoryless : public source<S, C> {
private:
   /*! \name Internal representation */
   libbase::randgen r; //!< Data sequence generator
   libbase::vector<float> cpt; //!< Cumulative symbol probability table
   // @}
private:
   /*! \name Internal functions */
   //! Obtain cumulative probabilities from symbol probabilities
   libbase::vector<float> to_cumulative(libbase::vector<float> symbol_probabilities) const;
   //! Obtain symbol probabilities from cumulative probabilities
   libbase::vector<float> to_probabilities(libbase::vector<float> cpt) const;
   // @}
public:
   //! Default constructor
   memoryless()
      {
      }
   //! Main constructor
   memoryless(libbase::vector<float> symbol_probabilities)
      {
      cpt = to_cumulative(symbol_probabilities);
      }

   //! Generate a single source element
   S generate_single()
      {
      const float p = r.fval_halfopen();
      int value = cpt.size() - 1;
      while(value > 0 && p < cpt(value-1))
         value--;
      return value;
      }

   //! Seeds any random generators from a pseudo-random sequence
   void seedfrom(libbase::random& r)
      {
      this->r.seed(r.ival());
      }

   //! Description
   std::string description() const
      {
      std::ostringstream sout;
      sout << "Memoryless source [p=";
      libbase::vector<float> symbol_probabilities = to_probabilities(cpt);
      symbol_probabilities.serialize(sout, ',');
      sout << "]";
      return sout.str();
      }

   // Serialization Support
DECLARE_SERIALIZER(memoryless)
};

} // end namespace

#endif

