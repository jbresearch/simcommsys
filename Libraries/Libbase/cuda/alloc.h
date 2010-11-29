#ifndef __cuda_alloc_h
#define __cuda_alloc_h

namespace cuda {

/*!
 * \brief   Allocator for host pinned (non-pageable) memory.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision: 3739 $
 * - $Date: 2010-06-17 15:32:37 +0100 (Thu, 17 Jun 2010) $
 * - $Author: jabriffa $
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
