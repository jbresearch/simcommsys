/*!
 \file

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$
 */

#include "commsys_errorrates.h"
#include "fsm.h"
#include "itfunc.h"
#include <sstream>

namespace libcomm {

/*!
 \brief Count the number of bit errors in the last encode/decode cycle
 \return Error count in bits
 */
int commsys_errorrates::countbiterrors(const libbase::vector<int>& source,
      const libbase::vector<int>& decoded) const
   {
   assert(source.size() == get_symbolsperblock());
   assert(decoded.size() == get_symbolsperblock());
   int biterrors = 0;
   for (int t = 0; t < get_symbolsperblock(); t++)
      {
      assert(source(t) != fsm::tail);
      biterrors += libbase::weight(source(t) ^ decoded(t));
      }
   return biterrors;
   }

/*!
 \brief Count the number of symbol errors in the last encode/decode cycle
 \return Error count in symbols
 */
int commsys_errorrates::countsymerrors(const libbase::vector<int>& source,
      const libbase::vector<int>& decoded) const
   {
   assert(source.size() == get_symbolsperblock());
   assert(decoded.size() == get_symbolsperblock());
   int symerrors = 0;
   for (int t = 0; t < get_symbolsperblock(); t++)
      {
      assert(source(t) != fsm::tail);
      if (source(t) != decoded(t))
         symerrors++;
      }
   return symerrors;
   }

/*!
 \brief Update result set
 \param[out] result   Vector containing the set of results to be updated
 \param[in]  i        Iteration just performed
 \param[in]  source   Source data sequence
 \param[in]  decoded  Decoded data sequence

 Results are organized as (symbol,frame) error count, repeated for
 every iteration that needs to be performed. Eventually these will be
 divided by the respective multiplicity to get the average error rates.
 */
void commsys_errorrates::updateresults(libbase::vector<double>& result,
      const int i, const libbase::vector<int>& source, const libbase::vector<
            int>& decoded) const
   {
   assert(i >= 0 && i < get_iter());
   // Count errors
   int symerrors = countsymerrors(source, decoded);
   // Estimate the BER, SER, FER
   result(2 * i + 0) += symerrors;
   result(2 * i + 1) += symerrors ? 1 : 0;
   }

/*!
 \copydoc experiment::get_multiplicity()

 Since results are organized as (symbol,frame) error count, repeated for
 every iteration, the multiplicity is respectively the number of symbols
 and the number of frames (=1) per sample.
 */
int commsys_errorrates::get_multiplicity(int i) const
   {
   assert(i >= 0 && i < count());
   if (i % 2 == 0)
      return get_symbolsperblock();
   return 1;
   }

/*!
 \copydoc experiment::result_description()

 The description is a string XER_Y, where 'X' is S,F to indicate
 symbol or frame error rates respectively. 'Y' is the iteration,
 starting at 1.
 */
std::string commsys_errorrates::result_description(int i) const
   {
   assert(i >= 0 && i < count());
   std::ostringstream sout;
   if (i % 2 == 0)
      sout << "SER_";
   else
      sout << "FER_";
   sout << (i / 2) + 1;
   return sout.str();
   }

} // end namespace
