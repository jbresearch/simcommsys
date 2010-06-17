#ifndef __aligned_allocator_h
#define __aligned_allocator_h

#include "config.h"
#include <memory>

namespace libbase {

/*!
 * \brief   Allocator that guarantees aligned memory.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

// Aligned memory allocator

template <class T, size_t alignment>
class aligned_allocator : public std::allocator<T> {
private:
   typedef std::allocator<T> base;
public:
   typedef typename base::pointer pointer;
   typedef typename base::size_type size_type;

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
