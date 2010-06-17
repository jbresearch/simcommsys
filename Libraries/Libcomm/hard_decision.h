#ifndef __hard_decision_h
#define __hard_decision_h

#include "config.h"
#include "vector.h"
#include "matrix.h"

namespace libcomm {

template <class dbl>
class basic_hard_decision {
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   // @}
public:
   int operator()(const array1d_t& ri)
      {
      // Inherit size
      const int K = ri.size();
      assert(K > 0);
      // Process
      int decoded = 0;
      for (int i = 1; i < K; i++)
         if (ri(i) > ri(decoded))
            decoded = i;
      return decoded;
      }
};

template <template <class > class C, class dbl>
class hard_decision : public basic_hard_decision<dbl> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   // @}
public:
   int operator()(const C<array1d_t>& ri, C<int>& decoded);
};

template <class dbl>
class hard_decision<libbase::vector, dbl> : public basic_hard_decision<dbl> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   // @}
public:
   /*!
    * \brief Hard decision on soft information
    * \param[in]  ri       Likelihood table for input symbols at every timestep
    * \param[out] decoded  Sequence of the most likely input symbols at every
    * timestep
    *
    * Decide which input sequence was most probable.
    */
   void operator()(const libbase::vector<array1d_t>& ri,
         libbase::vector<int>& decoded)
      {
      // Determine sizes from input matrix
      const int tau = ri.size();
      assert(tau > 0);
#ifndef NDEBUG
      const int K = ri(0).size();
#endif
      // Initialise result vector
      decoded.init(tau);
      // Determine most likely symbol at every timestep
      for (int t = 0; t < tau; t++)
         {
         assert(ri(t).size() == K);
         decoded(t) = basic_hard_decision<dbl>::operator()(ri(t));
         }
      }
};

template <class dbl>
class hard_decision<libbase::matrix, dbl> : public basic_hard_decision<dbl> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   // @}
public:
   /*!
    * \brief Hard decision on soft information
    * \param[in]  ri       Likelihood table for input symbols at every timestep
    * \param[out] decoded  Sequence of the most likely input symbols at every
    * timestep
    *
    * Decide which input sequence was most probable.
    */
   void operator()(const libbase::matrix<array1d_t>& ri,
         libbase::matrix<int>& decoded)
      {
      // Determine sizes from input matrix
      const int rows = ri.size().rows();
      const int cols = ri.size().cols();
      assert(rows > 0 && cols > 0);
#ifndef NDEBUG
      const int K = ri(0, 0).size();
#endif
      // Initialise result vector
      decoded.init(rows, cols);
      // Determine most likely symbol at every timestep
      for (int i = 0; i < rows; i++)
         for (int j = 0; j < cols; j++)
            {
            assert(ri(i, j).size() == K);
            decoded(i, j) = basic_hard_decision<dbl>::operator()(ri(i, j));
            }
      }
};

} // end namespace

#endif
