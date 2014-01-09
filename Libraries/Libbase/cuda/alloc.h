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

#ifndef __cuda_alloc_h
#define __cuda_alloc_h

namespace cuda {

/*!
 * \brief   Allocator for host pinned (non-pageable) memory.
 * \author  Johann Briffa
 */

template <class T>
class pinned_allocator : public std::allocator<T> {
private:
   typedef std::allocator<T> Base;
public:
   typedef typename Base::pointer pointer;
   typedef typename Base::size_type size_type;

   template <class U>
   struct rebind {
      typedef pinned_allocator<U> other;
   };

#ifdef __CUDACC__
   pointer allocate(size_type n, std::allocator<void>::const_pointer hint = 0)
      {
      void *p;
      if (cudaMallocHost(&p, n * sizeof(T)) == cudaSuccess)
         return (pointer) p;
      throw std::bad_alloc();
      }

   void deallocate(pointer p, size_type n)
      {
      if (cudaFreeHost(p) == cudaSuccess)
         return;
      throw std::bad_alloc();
      }
#endif
};

} // end namespace

#endif
