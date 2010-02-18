#ifndef LEVENSHTEIN_H_
#define LEVENSHTEIN_H_

#include "config.h"
#include "vector.h"
#include "matrix.h"

namespace libbase {

/*!
 * \brief   Compute Levenshtein Distance
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * A method that computes the Levenshtein distance between two sequences.
 * Templatized for any type for which a definition of equality exists.
 */

template <class T>
int levenshtein(const vector<T>& s, const vector<T>& t)
   {
   const int m = s.size();
   const int n = t.size();

   // d is a table with m+1 rows and n+1 columns
   matrix<int> d(m + 1, n + 1);

   // initialize values on edges
   for (int i = 0; i <= m; i++)
      d(i, 0) = i; // deletions
   for (int j = 0; j <= n; j++)
      d(0, j) = j; // insertion

   // fill in the rest of the table
   for (int j = 0; j < n; j++)
      for (int i = 0; i < m; i++)
         {
         if (s(i) == t(j))
            d(i + 1, j + 1) = d(i, j);
         else
            d(i + 1, j + 1) = std::min(std::min(//
                  d(i, j + 1) + 1, // deletion
                  d(i + 1, j) + 1), // insertion
                  d(i, j) + 1); // substitution
         }

   return d(m, n);
   }

} // end namespace

#endif /* LEVENSHTEIN_H_ */
