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

#ifndef __errors_levenshtein_h
#define __errors_levenshtein_h

#include "config.h"
#include "errors_hamming.h"

namespace libcomm {

/*!
 * \brief   CommSys Results - SER (Hamming & Levenshtein), FER.
 * \author  Johann Briffa
 *
 * Implements error rate calculators for SER (using both Hamming and
 * Levenshtein distances) and FER.
 */
class errors_levenshtein : public errors_hamming {
public:
   /*! \name Public interface */
   void updateresults(libbase::vector<double>& result, const libbase::vector<
         int>& source, const libbase::vector<int>& decoded) const;
   /*! \copydoc experiment::count()
    * We count the number of symbol errors using Hamming and Levenshtein
    * metrics, as well as the number of frame errors.
    */
   int count() const
      {
      return 3;
      }
   /*! \copydoc experiment::get_multiplicity()
    *
    * Since results are organized as (symbol_hamming, symbol_levenshtein,frame)
    * error count, the multiplicity is respectively the number of symbols
    * (twice) and the number of frames (=1) per sample.
    *
    * \warning In the case of Levenshtein distance, it is not clear how the
    * multiplicity should be computed.
    */
   int get_multiplicity(int i) const
      {
      assert(i >= 0 && i < count());
      switch (i)
         {
         case 0:
         case 1:
            return get_symbolsperblock();
         case 2:
            return 1;
         }
      return -1; // This should never happen
      }

   /*! \copydoc experiment::result_description()
    *
    * The description is a string of SER,LD,FER to indicate symbol error rate
    * (Hamming distance), Levenshtein distance, or frame error rate
    * respectively.
    */
   std::string result_description(int i) const
      {
      assert(i >= 0 && i < count());
      switch (i)
         {
         case 0:
            return "SER";
         case 1:
            return "LD";
         case 2:
            return "FER";
         }
      return ""; // This should never happen
      }
   // @}
};

} // end namespace

#endif
