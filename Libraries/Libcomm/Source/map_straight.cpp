/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "map_straight.h"
#include <stdlib.h>
#include <sstream>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Matrix: show input/output sizes on transform/inverse
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*** Vector Specialization ***/

using libbase::vector;

// Interface with mapper

/*! \copydoc mapper::setup()

   \note Each encoder output must be represented by an integral number of
         modulation symbols
*/
template <class dbl>
void map_straight<vector,dbl>::setup()
   {
   s1 = get_rate(M, N);
   s2 = get_rate(M, S);
   upsilon = size.x*s1/s2;
   assertalways(size.x*s1 == upsilon*s2);
   }

template <class dbl>
void map_straight<vector,dbl>::dotransform(const array1i_t& in, array1i_t& out) const
   {
   assertalways(in.size() == This::input_block_size());
   // Initialize results vector
   out.init(This::output_block_size());
   // Modulate encoded stream (least-significant first)
   for(int t=0, k=0; t<size.x; t++)
      for(int i=0, x = in(t); i<s1; i++, k++, x /= M)
         out(k) = x % M;
   }

template <class dbl>
void map_straight<vector,dbl>::doinverse(const array1vd_t& pin, array1vd_t& pout) const
   {
   // Confirm modulation symbol space is what we expect
   assertalways(pin.size() > 0);
   assertalways(pin(0).size() == M);
   // Confirm input sequence to be of the correct length
   assertalways(pin.size() == This::output_block_size());
   // Initialize results vector
   pout.init(upsilon);
   for(int t=0; t<upsilon; t++)
      pout(t).init(S);
   // Get the necessary data from the channel
   for(int t=0; t<upsilon; t++)
      for(int x=0; x<S; x++)
         {
         pout(t)(x) = 1;
         for(int i=0, thisx = x; i<s2; i++, thisx /= M)
            pout(t)(x) *= pin(t*s2+i)(thisx % M);
         }
   }

// Description

template <class dbl>
std::string map_straight<vector,dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Straight Mapper (Vector)";
   return sout.str();
   }

// Serialization Support

template <class dbl>
std::ostream& map_straight<vector,dbl>::serialize(std::ostream& sout) const
   {
   return sout;
   }

template <class dbl>
std::istream& map_straight<vector,dbl>::serialize(std::istream& sin)
   {
   return sin;
   }

/*** Matrix Specialization ***/

using libbase::matrix;

// Interface with mapper

/*! \copydoc mapper::setup()

   \note Symbol alphabets must be the same size
*/
template <class dbl>
void map_straight<matrix,dbl>::setup()
   {
   assertalways(M == N);
   assertalways(M == S);
   }

template <class dbl>
void map_straight<matrix,dbl>::dotransform(const array2i_t& in, array2i_t& out) const
   {
   assertalways(in.size() == This::input_block_size());
   // Initialize results matrix
   out.init(This::output_block_size());
#if DEBUG>=2
   libbase::trace << "DEBUG (map_straight): Transform ";
   libbase::trace << in.xsize() << "x" << in.ysize() << " to ";
   libbase::trace << out.xsize() << "x" << out.ysize() << "\n";
#endif
   // Map encoded stream (row-major order)
   int ii=0, jj=0;
   for(int i=0; i<in.xsize(); i++)
      for(int j=0; j<in.ysize(); j++)
         {
         out(ii,jj) = in(i,j);
         jj++;
         if(jj >= out.ysize())
            {
            jj = 0;
            ii++;
            }
         }
   }

template <class dbl>
void map_straight<matrix,dbl>::doinverse(const array2vd_t& pin, array2vd_t& pout) const
   {
   // Confirm modulation symbol space is what we expect
   assertalways(pin.size() > 0);
   assertalways(pin(0,0).size() == M);
   // Confirm input sequence to be of the correct length
   assertalways(pin.size() == This::output_block_size());
   // Initialize results vector
   pout.init(This::input_block_size());
#if DEBUG>=2
   libbase::trace << "DEBUG (map_straight): Inverse ";
   libbase::trace << pin.xsize() << "x" << pin.ysize() << " to ";
   libbase::trace << pout.xsize() << "x" << pout.ysize() << "\n";
#endif
   // Map channek receiver information (row-major order)
   int ii=0, jj=0;
   for(int i=0; i<pin.xsize(); i++)
      for(int j=0; j<pin.ysize(); j++)
         {
         pout(ii,jj) = pin(i,j);
         jj++;
         if(jj >= pout.ysize())
            {
            jj = 0;
            ii++;
            }
         }
   }

// Description

template <class dbl>
std::string map_straight<matrix,dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Straight Mapper (Matrix ";
   sout << size_out.x << "x" << size_out.y << ")";
   return sout.str();
   }

// Serialization Support

template <class dbl>
std::ostream& map_straight<matrix,dbl>::serialize(std::ostream& sout) const
   {
   sout << size_out.x << '\t' << size_out.y << '\n';
   return sout;
   }

template <class dbl>
std::istream& map_straight<matrix,dbl>::serialize(std::istream& sin)
   {
   sin >> size_out.x >> size_out.y;
   return sin;
   }

}; // end namespace

// Explicit Realizations

#include "logrealfast.h"

namespace libcomm {

using libbase::serializer;
using libbase::logrealfast;

/*** Vector Specialization ***/

template class map_straight<vector>;
template <>
const serializer map_straight<vector>::shelper("mapper", "map_straight<vector>", map_straight<vector>::create);

template class map_straight<vector,float>;
template <>
const serializer map_straight<vector,float>::shelper("mapper", "map_straight<vector,float>", map_straight<vector,float>::create);

template class map_straight<vector,logrealfast>;
template <>
const serializer map_straight<vector,logrealfast>::shelper("mapper", "map_straight<vector,logrealfast>", map_straight<vector,logrealfast>::create);

/*** Matrix Specialization ***/

template class map_straight<matrix>;
template <>
const serializer map_straight<matrix>::shelper("mapper", "map_straight<matrix>", map_straight<matrix>::create);

template class map_straight<matrix,float>;
template <>
const serializer map_straight<matrix,float>::shelper("mapper", "map_straight<matrix,float>", map_straight<matrix,float>::create);

template class map_straight<matrix,logrealfast>;
template <>
const serializer map_straight<matrix,logrealfast>::shelper("mapper", "map_straight<matrix,logrealfast>", map_straight<matrix,logrealfast>::create);

}; // end namespace
