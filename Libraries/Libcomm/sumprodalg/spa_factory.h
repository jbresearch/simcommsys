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

#ifndef SPA_FACTORY_H_
#define SPA_FACTORY_H_
#include "alist.h"
#include "gf.h"
#include "sum_prod_alg_inf.h"
#ifndef USE_CUDA
#    include "sumprodalg/impl/sum_prod_alg_gdl.h"
#else
#    include "sumprodalg/impl/sum_prod_alg_gdl_cuda.h"
#endif
#include "sumprodalg/impl/sum_prod_alg_trad.h"

#include "logrealfast.h"
#include <memory>
#include <string>

namespace libcomm
{
/*! \brief factory to return the desired SPA implementation
 * This factory allows the user to choose the SPA implementation
 * required for the code. Three choices are currently supported:
 * trad, gdl and gdl_cuda
 * trad is computationally expensive but easy to understand
 * gdl uses Fast Hadamard/Fourier Transforms to speed up the
 * computations.
 * gdl_cuda is an optimized port of gdl to CUDA C/C++
 */
template <class GF_q, class real = double>
class spa_factory
{
public:
    /*! \name Type definitions */
    typedef libbase::vector<int> array1i_t;
    typedef libbase::vector<array1i_t> array1vi_t;

public:
    /*!\brief return an instance of the SPA algorithm
     *
     */
    static std::shared_ptr<sum_prod_alg_inf<GF_q, real>>
    get_spa(const std::string type, const libbase::alist<GF_q> pchk_matrix)
    {
        if ("trad" == type) {
            return std::make_shared<sum_prod_alg_trad<GF_q, real>>(pchk_matrix);
        } else if ("gdl" == type) {
#ifndef USE_CUDA
            return std::make_shared<sum_prod_alg_gdl<GF_q, real>>(pchk_matrix);
#else
            return std::make_shared<sum_prod_alg_gdl_cuda<GF_q, real>>(
                pchk_matrix);
#endif
        } else {
            std::string error_msg(type + " is not a valid SPA type");
            failwith(error_msg.c_str());
            // appease compiler
            return nullptr;
        }
    }
};

} // namespace libcomm

#endif /* SPA_FACTORY_H_ */
