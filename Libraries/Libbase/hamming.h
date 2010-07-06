#ifndef HAMMING_H_
#define HAMMING_H_

#include "config.h"
#include "vector.h"
#include "matrix.h"

namespace libbase {

/*!
 * \brief   Compute Hamming Distance
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * A method that computes the Hamming distance between two sequences.
 * Templatized for any type for which a definition of equality exists.
 */

template <class T>
int hamming(const vector<T>& s, const vector<T>& t)
   {
   const int m = s.size();
   const int n = t.size();

   assertalways(m == n);

   // initialize distance
   int d = 0;

   // fill in the rest of the table
   for (int i = 0; i < n; i++)
      {
      if (s(i) == t(i))
         continue;
      else
         d++;
      }

   return d;
   }

} // end namespace

#endif /* HAMMING_H_ */
