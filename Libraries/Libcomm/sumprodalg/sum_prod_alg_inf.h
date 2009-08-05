/*
 * sum_prod_alg_inf.h
 *
 *  Created on: 21 Jul 2009
 *      Author: swesemeyer
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
};

}

#endif /* SUM_PROD_ALG_INF_H_ */
