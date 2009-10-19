/*
 * spa_factory.h
 *
 *  Created on: 22 Jul 2009
 *      Author: swesemeyer
 */

#ifndef SPA_FACTORY_H_
#define SPA_FACTORY_H_
#include "sum_prod_alg_inf.h"
#include "sumprodalg/impl/sum_prod_alg_trad.h"
#include "sumprodalg/impl/sum_prod_alg_gdl.h"
#include "gf.h"
#include "matrix.h"

#include "boost/shared_ptr.hpp"
using boost::shared_ptr;

#include <string>
#include "logrealfast.h"
using std::string;

namespace libcomm {
/*! \brief factory to return the desired SPA implementation
 * This factory allows the user to choose the SPA implementation
 * required for the code. Two choices are currently supported:
 * trad and gdl
 * trad is computationally expensive but easy to understand
 * gdl uses Fast Hadamard/Fourier Transforms to speed up the
 * computations.
 */
template <class GF_q, class real = double>
class spa_factory {
public:
   /*! \name Type definitions */
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<array1i_t> array1vi_t;

public:
   /*!\brief return an instance of the SPA algorithm
    *
    */
   static shared_ptr<sum_prod_alg_inf<GF_q, real> > get_spa(
         const std::string type, int n, int m,
         const array1vi_t& non_zero_col_pos,
         const array1vi_t& non_zero_row_pos, const matrix<GF_q> pchk_matrix)
      {
      shared_ptr<sum_prod_alg_inf<GF_q, real> > spa_ptr;
      if ("trad" == type)
         {
         spa_ptr = shared_ptr<sum_prod_alg_inf<GF_q, real> > (
               new sum_prod_alg_trad<GF_q, real> (n, m, non_zero_col_pos,
                     non_zero_row_pos, pchk_matrix));
         }
      else if ("gdl" == type)
         {
         spa_ptr = shared_ptr<sum_prod_alg_inf<GF_q, real> > (
               new sum_prod_alg_gdl<GF_q, real> (n, m, non_zero_col_pos,
                     non_zero_row_pos, pchk_matrix));
         }
      else
         {
         string error_msg(type + " is not a valid SPA type");
         failwith(error_msg.c_str());
         }
      return spa_ptr;
      }
};

}

#endif /* SPA_FACTORY_H_ */

//Explicit realisations
#include "mpreal.h"

namespace libcomm {
using libbase::mpreal;

template class spa_factory<gf<1, 0x3> > ;
template class spa_factory<gf<2, 0x7> > ;
template class spa_factory<gf<3, 0xB> > ;
template class spa_factory<gf<3, 0xB> , mpreal> ;
template class spa_factory<gf<4, 0x13> > ;
template class spa_factory<gf<4, 0x13> , mpreal> ;
template class spa_factory<gf<5, 0x25> > ;
template class spa_factory<gf<6, 0x43> > ;
template class spa_factory<gf<7, 0x89> > ;
template class spa_factory<gf<8, 0x11D> > ;
template class spa_factory<gf<9, 0x211> > ;
template class spa_factory<gf<10, 0x409> > ;

}
