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

#ifndef __map_concatenated_h
#define __map_concatenated_h

#include "mapper.h"
#include <list>

namespace libcomm {

/*!
 * \brief   Mapper representing a concatenation of mappers.
 * \author  Johann Briffa
 *
 * This class defines a concatenation of mappers.
 */

template <template <class > class C = libbase::vector, class dbl = double>
class map_concatenated : public mapper<C, dbl> {
private:
   // Shorthand for class hierarchy
   typedef mapper<C, dbl> Base;
   typedef map_concatenated<C, dbl> This;
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   typedef std::list<mapper<C, dbl> *> mapper_list_t;
   typedef std::list<int> interface_list_t;
   // @}

private:
   /*! \name Internal object representation */
   mapper_list_t mapper_list;
   interface_list_t interface_list;
   libbase::size_type<C> osize; //!< Output block size in symbols
   // @}

protected:
   /*! \name Internal operations */
   //! Invariance test
   void test_invariant() const
      {
      // The tests below assume a properly set up and usable system
      assert(mapper_list.size() >= 1);
      assert(interface_list.size() == mapper_list.size() - 1);
      }
   void free()
      {
      // Destroy mappers
      for(typename mapper_list_t::iterator it = mapper_list.begin(); it != mapper_list.end(); it++)
         delete *it;
      mapper_list.clear();
      // Delete list of interface alphabet sizes
      interface_list.clear();
      }
   // @}
   // Interface with mapper
   void setup();
   void advance() const
      {
      test_invariant();
      // Advance mappers
      for(typename mapper_list_t::const_iterator it = mapper_list.begin(); it != mapper_list.end(); it++)
         (*it)->advance_always();
      }
   void status_changed() const
      {
      test_invariant();
      // Advance mappers
      for(typename mapper_list_t::const_iterator it = mapper_list.begin(); it != mapper_list.end(); it++)
         {
         if (this->dirty)
            (*it)->mark_as_dirty();
         else
            (*it)->mark_as_clean();
         }
      }
   void dotransform(const C<int>& in, C<int>& out) const;
   void dotransform(const C<array1d_t>& pin, C<array1d_t>& pout) const;
   void doinverse(const C<array1d_t>& pin, C<array1d_t>& pout) const;

public:
   // Setup functions
   void seedfrom(libbase::random& r)
      {
      test_invariant();
      // Seed mappers
      for(typename mapper_list_t::iterator it = mapper_list.begin(); it != mapper_list.end(); it++)
         (*it)->seedfrom(r);
      }

   // Informative functions
   libbase::size_type<libbase::vector> output_block_size() const
      {
      return osize;
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(map_concatenated)
};

} // end namespace

#endif
