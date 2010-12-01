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
 * 
 * \section svn Version Control
 * - $Id$
 */

#ifndef __bitfield_cuda_h
#define __bitfield_cuda_h

#include "config.h"
#include "cuda_assert.h"

namespace cuda {

/*!
 * \brief   Bitfield (register of a set size) for CUDA.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

class bitfield {
private:
   libbase::int32u field; //!< bit field value
   int bits; //!< number of bits
private:
   //! Check that given field size is representable by class
#ifdef __CUDACC__
   __device__ __host__
#endif
   static void check_fieldsize(int b)
      {
#ifndef __CUDA_ARCH__ // Host code path
      assertalways(b>=0 && b<=32);
#endif
      }
   //! Get binary mask covering bitfield
#ifdef __CUDACC__
   __device__ __host__
#endif
   libbase::int32u mask() const
      {
      return bits == 32 ? 0xffffffff : ((1L << bits) - 1L);
      }
public:
   /*! \name Constructors / Destructors */
   //! Default constructor
#ifdef __CUDACC__
   __device__ __host__
#endif
   explicit bitfield(const int bits = 0) :
      field(0), bits(bits)
      {
      }
   //! Constructor to directly convert an integer at a specified width
#ifdef __CUDACC__
   __device__ __host__
#endif
   bitfield(const libbase::int32u field, const int bits) :
      field(field), bits(bits)
      {
      }
   // @}

   /*! \name Type conversion */
#ifdef __CUDACC__
   __device__ __host__
#endif
   operator libbase::int32u() const
      {
      return field;
      }
   // @}

   // Field size methods
#ifdef __CUDACC__
   __device__ __host__
#endif
   int size() const
      {
      return bits;
      }
#ifdef __CUDACC__
   __device__ __host__
#endif
   void init(const int b)
      {
      check_fieldsize(b);
      bits = b;
      field &= mask();
      }

   // Copy and Assignment
#ifdef __CUDACC__
   __device__ __host__
#endif
   bitfield& operator=(const bitfield& x)
      {
      bits = x.bits;
      field = x.field;
      return *this;
      }
#ifdef __CUDACC__
   __device__ __host__
#endif
   bitfield& operator=(const libbase::int32u x)
      {
      cuda_assert((x & ~mask()) == 0);
      field = x;
      return *this;
      }

   // Partial extraction
#ifdef __CUDACC__
   __device__ __host__
#endif
   bitfield extract(const int hi, const int lo) const
      {
      bitfield c;
      cuda_assert(hi < bits && lo >= 0 && lo <= hi);
      c.bits = hi - lo + 1;
      c.field = (field >> lo) & c.mask();
      return c;
      }
#ifdef __CUDACC__
   __device__ __host__
#endif
   bitfield extract(const int b) const
      {
      return extract(b, b);
      }
   // Indexed access
#ifdef __CUDACC__
   __device__ __host__
#endif
   bool operator()(const int b) const
      {
      cuda_assert(b < bits && b >= 0);
      return (field >> b) & 1;
      }

   // Bit-reversal method
#ifdef __CUDACC__
   __device__ __host__
#endif
   bitfield reverse() const
      {
      bitfield result;
      for (int i = 0; i < bits; i++)
         result = result + extract(i);
      return result;
      }

   // Logical operators - OR, AND, XOR
#ifdef __CUDACC__
   __device__ __host__
#endif
   bitfield operator|(const bitfield& x) const
      {
      bitfield y = *this;
      y |= x;
      return y;
      }
#ifdef __CUDACC__
   __device__ __host__
#endif
   bitfield operator&(const bitfield& x) const
      {
      bitfield y = *this;
      y &= x;
      return y;
      }
#ifdef __CUDACC__
   __device__ __host__
#endif
   bitfield operator^(const bitfield& x) const
      {
      bitfield y = *this;
      y ^= x;
      return y;
      }
#ifdef __CUDACC__
   __device__ __host__
#endif
   bitfield& operator|=(const bitfield& x)
      {
      cuda_assert(bits == x.bits);
      field |= x.field;
      return *this;
      }
#ifdef __CUDACC__
   __device__ __host__
#endif
   bitfield& operator&=(const bitfield& x)
      {
      cuda_assert(bits == x.bits);
      field &= x.field;
      return *this;
      }
#ifdef __CUDACC__
   __device__ __host__
#endif
   bitfield& operator^=(const bitfield& x)
      {
      cuda_assert(bits == x.bits);
      field ^= x.field;
      return *this;
      }

   // Convolution operator
#ifdef __CUDACC__
   __device__ __host__
#endif
   bitfield operator*(const bitfield& x) const
      {
      cuda_assert(this->bits == x.bits);
      bitfield y;
      y.bits = 1;
      libbase::int32u r = this->field & x.field;
      for (int i = 0; i < this->bits; i++)
         if (r & (1 << i))
            y.field ^= 1;
      return y;
      }
   // Concatenation operator
#ifdef __CUDACC__
   __device__ __host__
#endif
   bitfield operator+(const bitfield& x) const
      {
      bitfield y;
      y.bits = this->bits + x.bits;
      y.field = (this->field << x.bits) | x.field;
      return y;
      }
   // Shift-register operators - sequence shift-in
#ifdef __CUDACC__
   __device__ __host__
#endif
   bitfield operator<<(const bitfield& x) const
      {
      bitfield y;
      y.bits = this->bits;
      y.field = (this->field << x.bits) | x.field;
      y.field &= y.mask();
      return y;
      }
#ifdef __CUDACC__
   __device__ __host__
#endif
   bitfield operator>>(const bitfield& x) const
      {
      bitfield y;
      y.bits = x.bits;
      y.field = (this->field << (x.bits - this->bits))
            | (x.field >> this->bits);
      y.field &= y.mask();
      return y;
      }
   // Shift-register operators - zero shift-in
#ifdef __CUDACC__
   __device__ __host__
#endif
   bitfield operator<<(const int x) const
      {
      bitfield c = *this;
      c <<= x;
      return c;
      }
#ifdef __CUDACC__
   __device__ __host__
#endif
   bitfield operator>>(const int x) const
      {
      bitfield c = *this;
      c >>= x;
      return c;
      }
#ifdef __CUDACC__
   __device__ __host__
#endif
   bitfield& operator<<=(const int x)
      {
      field <<= x;
      field &= mask();
      return *this;
      }
#ifdef __CUDACC__
   __device__ __host__
#endif
   bitfield& operator>>=(const int x)
      {
      if (x >= 32) // avoid a subtle bug in gcc (or the Pentium, not sure which...)
         field = 0;
      else
         {
         field >>= x;
         field &= mask();
         }
      return *this;
      }
};

} // end namespace

#endif
