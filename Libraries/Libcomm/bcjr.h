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

#ifndef __bcjr_h
#define __bcjr_h

#include "config.h"
#include "vector.h"
#include "matrix.h"
#include "matrix3.h"

#include "sigspace.h"
#include "fsm.h"

#include <cmath>
#include <iostream>
#include <fstream>

namespace libcomm {

/*!
 * \brief   Bahl-Cocke-Jelinek-Raviv (BCJR) decoding algorithm.
 * \author  Johann Briffa
 *
 * All internal metrics are held as type 'real', which is user-defined. This
 * allows internal working at any required level of accuracy. This is required
 * because the internal matrics have a very wide dynamic range, which increases
 * exponentially with block size 'tau'. Actually, the required range is within
 * [1,0), but very large exponents are required.
 *
 * The second template class 'dbl', which defaults to 'double', allows other
 * numerical representations for externally-transferred statistics. This became
 * necessary for the parallel decoding structure, where the range of extrinsic
 * information is much larger than for serial decoding; furthermore, this range
 * increases with the number of iterations performed.
 *
 * The third template parameter 'norm', which defaults to false, is a flag to
 * enable conventional normalization of probabilities during forward and
 * backward recursion. This allows the use of double-precision representation
 * throughout the algorithm.
 *
 * \warning
 * - Static memory requirements:
 * sizeof(real)*(2*(tau+1)*M + tau*M*K + K + N) + sizeof(int)*(2*K+1)*M
 * - Dynamic memory requirements:
 * none
 *
 * \note Memory is only allocated in the first call to "decode". This is more
 * efficient for the parallel simulator strategy with a master which only
 * collects results.
 */

template <class real, class dbl = double, bool norm = false>
class bcjr {
public:
   /*! \name Type definitions */
   typedef libbase::vector<int> array1i_t;
   typedef libbase::matrix<int> array2i_t;
   typedef libbase::vector<dbl> array1d_t;
   typedef libbase::matrix<dbl> array2d_t;
   typedef libbase::matrix<real> array2r_t;
   typedef libbase::matrix3<real> array3r_t;
   // @}
private:
   /*! \name Internal variables */
   int tau; //!< Input block size in symbols (including tail)
   int K; //!< Input alphabet size
   int N; //!< Output alphabet size
   int M; //!< Number of encoder states
   bool initialised; //!< Flag to indicate when memory is allocated
   // @}
   /*! \name Working matrices */
   //! Forward recursion metric: alpha(t,m) = Pr{S(t)=m, Y(1..t)}
   array2r_t alpha;
   //! Backward recursion metric: beta(t,m) = Pr{Y(t+1..tau) | S(t)=m}
   array2r_t beta;
   //! Receiver metric: gamma(t-1,m',i) = Pr{S(t)=m(m',i), Y(t) | S(t-1)=m'}
   array3r_t gamma;
   // @}
   /*! \name Temporary (cache) matrices */
   //! lut_X(m,i) = encoder output, given state 'm' and input 'i'
   array2i_t lut_X;
   //! lut_m(m,i) = next state, given state 'm' and input 'i'
   array2i_t lut_m;
   // @}
private:
   /*! \name Internal methods */
   void allocate();
   real lambda(const int t, const int m);
   real sigma(const int t, const int m, const int i);
   void work_gamma(const array2d_t& R);
   void work_gamma(const array2d_t& R, const array2d_t& app);
   void work_alpha();
   void work_beta();
   void work_results(array2d_t& ri, array2d_t& ro);
   void work_results(array2d_t& ri);
   // @}
protected:
   // normalization function for derived classes
   static void normalize(array2d_t& r);
   // main initialization routine - constructor essentially just calls this
   void init(fsm& encoder, const int tau);
   // get start- and end-state probabilities
   array1d_t getstart() const;
   array1d_t getend() const;
   // set start- and end-state probabilities - equiprobable
   void setstart();
   void setend();
   // set start- and end-state probabilities - known state
   void setstart(int state);
   void setend(int state);
   // set start- and end-state probabilities - direct
   void setstart(const array1d_t& p);
   void setend(const array1d_t& p);
   // default constructor
   bcjr()
      {
      initialised = false;
      }
public:
   /*! \name Constructor & destructor */
   bcjr(fsm& encoder, const int tau)
      {
      init(encoder, tau);
      }
   // @}

   /*! \name Decode functions */
   void decode(const array2d_t& R, array2d_t& ri, array2d_t& ro);
   void decode(const array2d_t& R, const array2d_t& app, array2d_t& ri,
         array2d_t& ro);
   void fdecode(const array2d_t& R, array2d_t& ri);
   void fdecode(const array2d_t& R, const array2d_t& app, array2d_t& ri);
   // @}

   /*! \name Information functions */
   //! Number of defined states
   int num_states() const
      {
      return M;
      }
   //! Input alphabet size
   int num_input_symbols() const
      {
      return K;
      }
   //! Output alphabet size
   int num_output_symbols() const
      {
      return N;
      }
   //! Sequence length (number of time-steps)
   libbase::size_type<libbase::vector> block_size() const
      {
      return libbase::size_type<libbase::vector>(tau);
      }
   // @}
};

} // end namespace

#endif

