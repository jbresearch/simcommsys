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

#ifndef __aligned_allocator_h
#define __aligned_allocator_h

#include "config.h"
#include <memory>

namespace libbase {

/*!
 * \brief   Allocator that guarantees aligned memory.
 * \author  Johann Briffa
 */

template <class T, size_t alignment>
class aligned_allocator : public std::allocator<T> {
private:
   typedef std::allocator<T> Base;
public:
   typedef typename Base::pointer pointer;
   typedef typename Base::size_type size_type;

   template <class U>
   struct rebind {
      typedef aligned_allocator<U, alignment> other;
   };

   pointer allocate(size_type n, std::allocator<void>::const_pointer hint = 0)
      {
      void *p;
      if (posix_memalign(&p, alignment, n * sizeof(T)) == 0)
         {
         assert(isaligned(p, alignment));
         return (pointer) p;
         }
      throw std::bad_alloc();
      }

   void deallocate(pointer p, size_type n)
      {
      free(p);
      }
};

} // end namespace

#endif
