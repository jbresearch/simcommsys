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

#ifndef __group_h
#define __group_h

#include "config.h"

#include <iostream>

namespace libbase {

/*!
 * \brief   Group.
 * \author  Johann Briffa
 *
 * Class that represents the group. Used to save and load previously generated
 * groups.
 *
 * The class takes a BigInteger template parameter.
 *
 * \todo Add generation of a new group
 */

template <class BigInteger>
class group {
private:
   BigInteger p;
   BigInteger q;
   BigInteger g;

public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   group()
      {
      }
   explicit group(BigInteger p, BigInteger q, BigInteger g) :
      p(p), q(q), g(g)
      {
      }
   // @}

   void init(BigInteger p, BigInteger q, BigInteger g)
      {
      this->p = p;
      this->q = q;
      this->g = g;
      }

   /*! \name Getters */
   const BigInteger& get_p() const
      {
      return p;
      }
   const BigInteger& get_q() const
      {
      return q;
      }
   const BigInteger& get_g() const
      {
      return g;
      }
   // @}

   static BigInteger get_random_integer(BigInteger n)
      {
      BigInteger r;
      int maxbits = n.size();
      do
         {
         r.random(maxbits);
         } while (r >= n);
      return r;
      }

   BigInteger sample()
      {
      BigInteger r = get_random_integer(q);
      return g.pow_mod(r, p);
      }

   /*! \name Stream I/O */
   friend std::ostream& operator<<(std::ostream& sout, const group& x)
      {
      sout << x.p << std::endl;
      sout << x.q << std::endl;
      sout << x.g << std::endl;
      return sout;
      }

   friend std::istream& operator>>(std::istream& sin, group& x)
      {
      sin >> x.p;
      sin >> x.q;
      sin >> x.g;
      return sin;
      }
   // @}
};

} // end namespace

#endif
