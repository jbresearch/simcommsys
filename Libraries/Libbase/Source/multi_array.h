#ifndef __multi_array_h
#define __multi_array_h

#include "config.h"
#include <boost/multi_array.hpp>

namespace boost {

/*!
   \brief   Assignable version of Boost MultiArray.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

template<typename T, std::size_t NumDims>
class assignable_multi_array : public multi_array<T,NumDims> {
public:
   /*! \name Constructors / Destructors */
   explicit assignable_multi_array() :
      multi_array<T,NumDims>()
      {};
   explicit assignable_multi_array(const detail::multi_array::extent_gen<NumDims>& ranges) :
      multi_array<T,NumDims>(ranges)
      {};
   explicit assignable_multi_array(const assignable_multi_array& x) :
      multi_array<T,NumDims>(dynamic_cast< const multi_array<T,NumDims>& >(x))
      {};
   // @}
   /*! \name Assignment */
   assignable_multi_array& operator=(const assignable_multi_array& x)
      {
      if(!std::equal(this->shape(), this->shape()+this->num_dimensions(), x.shape()))
         {
         resize(x.extents());
         for(std::size_t i=0; i<NumDims; i++)
            libbase::trace << "Output Extent " << i << " = " << this->shape()[i] << "\n";
         }
      dynamic_cast< multi_array<T,NumDims>& >(*this) = dynamic_cast< const multi_array<T,NumDims>& >(x);
      return *this;
      }
   // @}
   /*! \name Informative functions */
   /*! \brief Get array extents description
      Returns an object describing the array extents, in a format suitable
      for use with resize().
   */
   detail::multi_array::extent_gen<NumDims> extents() const
      {
      typedef typename multi_array<T,NumDims>::extent_range extent_range;
      detail::multi_array::extent_gen<NumDims> extents_list;
      for(std::size_t i=0; i<NumDims; i++)
         {
         extents_list.ranges_[i] = extent_range(this->index_bases()[i], this->index_bases()[i]+this->shape()[i]);
         libbase::trace << "Input Extent " << i << " = " << this->shape()[i] << "\n";
         }
      return extents_list;
      }
   // @}
};

}; // end namespace

#endif
