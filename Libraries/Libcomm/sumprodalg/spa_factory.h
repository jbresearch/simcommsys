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
#include "gf.h"
#include "matrix.h"
#include "sum_prod_alg_inf.h"
#include "sumprodalg/impl/sum_prod_alg_gdl.h"
#include "sumprodalg/impl/sum_prod_alg_trad.h"

#include "logrealfast.h"
#include <memory>
#include <string>

namespace libcomm
{
/*! \brief factory to return the desired SPA implementation
 * This factory allows the user to choose the SPA implementation
 * required for the code. Two choices are currently supported:
 * trad and gdl
 * trad is computationally expensive but easy to understand
 * gdl uses Fast Hadamard/Fourier Transforms to speed up the
 * computations.
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
    get_spa(const std::string type,
            int n,
            int m,
            const array1vi_t& non_zero_col_pos,
            const array1vi_t& non_zero_row_pos,
            const libbase::matrix<GF_q> pchk_matrix)
    {
        std::shared_ptr<sum_prod_alg_inf<GF_q, real>> spa_ptr;
        if ("trad" == type) {
            spa_ptr = std::shared_ptr<sum_prod_alg_inf<GF_q, real>>(
                new sum_prod_alg_trad<GF_q, real>(
                    n, m, non_zero_col_pos, non_zero_row_pos, pchk_matrix));
        } else if ("gdl" == type) {
            spa_ptr = std::shared_ptr<sum_prod_alg_inf<GF_q, real>>(
                new sum_prod_alg_gdl<GF_q, real>(
                    n, m, non_zero_col_pos, non_zero_row_pos, pchk_matrix));
        } else {
            std::string error_msg(type + " is not a valid SPA type");
            failwith(error_msg.c_str());
        }
        return spa_ptr;
    }
};

} // namespace libcomm

#endif /* SPA_FACTORY_H_ */
