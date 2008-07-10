#ifndef __multi_array_h
#define __multi_array_h

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
protected:
   //template <std::size_t N>
   //detail::multi_array::extent_gen<N> gen_extents(detail::multi_array::extent_gen<NumDims-N> prev) const
   //   {
   //   assert(1 <= N && N <= NumDims);
   //   detail::multi_array::extent_gen<NumDims-N+1> next = prev[ shape()[NumDims-N] ];
   //   if(N == 1)
   //      return next;
   //   else
   //      return gen_extents<N-1>(next);
   //   }
public:
   /*! \name Constructors / Destructors */
   explicit assignable_multi_array() :
      multi_array<T,NumDims>()
      {};
   //template <class ExtentList>
   //explicit assignable_multi_array(const ExtentList& extents) :
   //   multi_array<T,NumDims>(extents)
   //   {};
   explicit assignable_multi_array(const detail::multi_array::extent_gen<NumDims>& ranges) :
      multi_array<T,NumDims>(ranges)
      {};
   // @}
   /*! \name The Big Three */
   explicit assignable_multi_array(const assignable_multi_array& x) :
      multi_array<T,NumDims>(dynamic_cast< const multi_array<T,NumDims>& >(x))
      {};
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
};

}; // end namespace

#endif
