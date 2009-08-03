/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "fsm.h"

namespace libcomm {

const int fsm::tail = -1;

// Helper functions

/*!
 * \brief Conversion from vector spaces to integer
 * \param[in] x Input in vector representation
 * \param[in] S Alphabet size for vector symbols
 * \return Value of \c x in integer representation
 *
 * Left-most register positions (ie. those closest to the input junction) are
 * represented by lower index positions, and get lower-order positions within
 * the integer representation.
 *
 * \todo check we are within the acceptable range for int representation
 */
int fsm::convert(const vector<int>& x, int S)
   {
   int nu = x.size();
   int y = 0;
   for (int i = nu - 1; i >= 0; i--)
      {
      y *= S;
      y += x(i);
      }
   return y;
   }

/*!
 * \brief Conversion from integer to vector space
 * \param[in] x Input in integer representation
 * \param[in] nu Length of vector representation
 * \param[in] S Alphabet size for vector symbols
 * \return Value of \c x in vector representation
 *
 * Left-most register positions (ie. those closest to the input junction) are
 * represented by lower index positions, and get lower-order positions within
 * the integer representation.
 */
vector<int> fsm::convert(int x, int nu, int S)
   {
   vector<int> y(nu);
   for (int i = 0; i < nu; i++)
      {
      y(i) = x % S;
      x /= S;
      }
   assert(x == 0);
   return y;
   }

// FSM state operations

void fsm::reset()
   {
   N = 0;
   }

void fsm::reset(libbase::vector<int> state)
   {
   N = 0;
   }

void fsm::resetcircular()
   {
   resetcircular(state(), N);
   }

// FSM operations

void fsm::advance(libbase::vector<int>& input)
   {
   N++;
   }

libbase::vector<int> fsm::step(libbase::vector<int>& input)
   {
   libbase::vector<int> op = output(input);
   advance(input);
   return op;
   }

} // end namespace
