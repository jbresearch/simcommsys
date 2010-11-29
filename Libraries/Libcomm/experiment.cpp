/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "experiment.h"

namespace libcomm {

using libbase::vector;

void experiment::prettyprint_results(std::ostream& sout) const
   {
   vector<double> result;
   vector<double> tolerance;
   estimate(result, tolerance);
   const int N = result.size();
   for (int i = 0; i < N; i++)
      {
      sout << result_description(i) << '\t';
      sout << result(i) << '\t';
      sout << "[+/- " << 100 * tolerance(i) / result(i) << "%]" << std::endl;
      }
   }

} // end namespace
