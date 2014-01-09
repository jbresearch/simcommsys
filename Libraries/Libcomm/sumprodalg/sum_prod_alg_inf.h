/*!
 * \file
 *
 * Copyright (c) 2010 Stephan Wesemeyer
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

#ifndef SUM_PROD_ALG_INF_H_
#define SUM_PROD_ALG_INF_H_
#include <string>
#include "vector.h"
#include "matrix.h"

namespace libcomm {
/*! \brief Sum Product Algorithm(SPA) interface
 * This is the interface that all SPA implementations need
 * to inherit from. It allows the SPA to be initialise with
 * the relevant probabilities.
 * Note that other initialisation parameters are passed in through
 * the constructor of the SPA implementation
 */
template <class GF_q, class real = double>
class sum_prod_alg_inf {
public:
   /*! \name Type definitions */
   typedef libbase::vector<real> array1d_t;
   typedef libbase::vector<array1d_t> array1vd_t;

   virtual ~sum_prod_alg_inf()
      {
      }

   /*! \brief initialise the SPA with the relevant probabilities
    *
    */
   virtual void spa_init(const array1vd_t& ptable)=0;
   /*! \brief carry out the SPA iteration
    *
    */
   virtual void spa_iteration(array1vd_t& ro)=0;
   /*! \brief return the type of SPA used
    *
    */
   virtual std::string spa_type()=0;

   /*! \brief set the way the algorithm should deal with
    * clipping, ie replacing probabilities below a certain value
    */

   virtual void set_clipping(std::string clipping_type, real almost_zero)=0;

   /*!\brief returns the type of clipping used
    *
    */
   virtual std::string get_clipping_type()=0;

   /*!\brief returns the value of almostzero used in the clipping method
    *
    */
   virtual real get_almostzero()=0;

   /*!\brief Perform the desired clipping
    *
    */
   virtual void perform_clipping(real& num)=0;
};

}

#endif /* SUM_PROD_ALG_INF_H_ */
