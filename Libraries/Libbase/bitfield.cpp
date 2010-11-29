/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "bitfield.h"

#include <cstdlib>
#include <string>

namespace libbase {

using std::cerr;

// Private functions

void bitfield::set_fromstring(const char *s)
   {
   bits = 0;
   field = 0;
   const char *p;
   for (p = s; *p == '1' || *p == '0'; p++)
      {
      field <<= 1;
      field |= (*p == '1');
      bits++;
      }
   // check there do not remain any invalid characters
   assertalways(*p == '\0');
   }

// Conversion operations

/*!
 * \brief Convert bitfield to a string representation
 */
std::string bitfield::asstring() const
   {
   std::string s;
   for (int i = bits - 1; i >= 0; i--)
      s += '0' + ((field >> i) & 1);
   return s;
   }

// Creation and Destruction

/*!
 * \brief Constructor that converts a vector of bits
 * 
 * Bits are held in the vector as low-order first; this means that the first
 * (index 0) element in the vector is the right-most (or least-significant) bit.
 * This convention is consistent with the convention used for bit indexing
 * using the [] operator, and also with that for converting vectors to integer
 * representation in fsm.
 */
bitfield::bitfield(const vector<bool>& v)
   {
   bits = v.size();
   check_fieldsize(bits);
   field = 0;
   for (int i = 0; i < bits; i++)
      field |= v(i) << i;
   }

/*!
 * \brief Convert bitfield to a vector representation
 * \sa bitfield()
 */
vector<bool> bitfield::asvector() const
   {
   vector<bool> result(bits);
   for (int i = 0; i < bits; i++)
      result(i) = ((field >> i) & 1);
   return result;
   }

} // end namespace
