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

#ifndef __erasable_h
#define __erasable_h

#include "config.h"

namespace libbase {

/*!
 * \brief   Erasable symbol - Templated base.
 * \author  Johann Briffa
 *
 * Implements the concept of a symbol that can be erased, where the symbol type
 * is specified as a template parameter. The symbol type must support the
 * following methods:
 * - Conversion from integer for values within range via constructor
 * - Conversion to integer using int() operator
 * - An elements() methods that returns the alphabet size
 */

template <class symbol>
class erasable {
private:
   /*! \name Object representation */
   //! Representation of this element
   symbol value;
   //! Flag indicating whether this element has been erased
   bool erased;
   // @}

public:
   /*! \name Constructors / Destructors */
   //! Principal constructor
   explicit erasable(int value = 0) :
         value(value), erased(false)
      {
      }
   explicit erasable(const symbol& value) :
         value(value), erased(false)
      {
      }
   // @}

   /*! \name Type conversion */
   operator int() const
      {
      assertalways(!erased);
      return int(value);
      }
   erasable& operator=(const int value)
      {
      this->value = symbol(value);
      this->erased = false;
      return *this;
      }
   operator symbol() const
      {
      assertalways(!erased);
      return value;
      }
   erasable& operator=(const symbol& value)
      {
      this->value = value;
      this->erased = false;
      return *this;
      }
   // @}

   /*! \name Erasure interface */
   void erase()
      {
      assertalways(!erased);
      erased = true;
      }
   bool is_erased() const
      {
      return erased;
      }
   // @}

   /*! \name Comparison operators */
   bool operator==(const erasable<symbol>& rhs) const
      {
      if (erased && rhs.erased)
         return true;
      if (!erased && !rhs.erased && value == rhs.value)
         return true;
      return false;
      }
   // @}

   /*! \name Class parameters */
   //! Number of elements in the finite alphabet
   static int elements()
      {
      return symbol::elements();
      }
   // @}
};

/*!
 * \brief   Erasable symbol - Bool specialization.
 * \author  Johann Briffa
 *
 * Implements the concept of a bool symbol that can be erased. Specialization
 * is necessary as the bool type does not support the required methods.
 */

template <>
class erasable<bool> {
private:
   /*! \name Object representation */
   //! Representation of this element
   bool value;
   //! Flag indicating whether this element has been erased
   bool erased;
   // @}

public:
   /*! \name Constructors / Destructors */
   //! Principal constructor
   explicit erasable(int value = 0) :
         erased(false)
      {
      assert(value >= 0 && value <= 1);
      this->value = value & 1;
      }
   explicit erasable(const bool& value) :
         value(value), erased(false)
      {
      }
   // @}

   /*! \name Type conversion */
   operator int() const
      {
      assertalways(!erased);
      return int(value);
      }
   erasable& operator=(const int value)
      {
      assert(value >= 0 && value <= 1);
      this->value = value & 1;
      this->erased = false;
      return *this;
      }
   operator bool() const
      {
      assertalways(!erased);
      return value;
      }
   erasable& operator=(const bool value)
      {
      assert(value >= 0 && value <= 1);
      this->value = value;
      this->erased = false;
      return *this;
      }
   // @}

   /*! \name Erasure interface */
   void erase()
      {
      assertalways(!erased);
      erased = true;
      }
   bool is_erased() const
      {
      return erased;
      }
   // @}

   /*! \name Comparison operators */
   bool operator==(const erasable<bool>& rhs) const
      {
      if (erased && rhs.erased)
         return true;
      if (!erased && !rhs.erased && value == rhs.value)
         return true;
      return false;
      }
   // @}

   /*! \name Class parameters */
   //! Number of elements in the finite alphabet
   static int elements()
      {
      return 2;
      }
   // @}
};

// Stream input / output

template <class symbol>
std::ostream& operator<<(std::ostream& s, const erasable<symbol>& x)
   {
   if (x.is_erased())
      s << "?";
   else
      s << symbol(x);
   return s;
   }

template <class symbol>
std::istream& operator>>(std::istream& s, erasable<symbol>& x)
   {
   char c;
   if (s.get(c))
      {
      if (c == '?')
         x.erase();
      else
         {
         s.putback(c);
         symbol temp;
         s >> temp;
         x = temp;
         }
      }
   return s;
   }

} // end namespace

#endif
