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

#include "sum_prod_alg_trad.h"

namespace libcomm
{

// Determine debug level:
// 1 - Normal debug output only
#ifndef NDEBUG
#    undef DEBUG
#    define DEBUG 1
#endif

template <class GF_q, class real>
void
sum_prod_alg_trad<GF_q, real>::spa_init(const array2d_t& recvd_probs)
{
    this->init_timer_with_variance("t_spa_iteration");

    // reset flag that informs us whether decoding has succeeded.
    this->decode_success = false;
    this->num_iters = 0;

    // initialise the marginal prob values
    int num_of_elements = GF_q::elements();
    real tmp_prob = real(0.0);
    real alpha = real(0.0);

    // ensure we don't have zero probabilities
    // and normalise the probs at the same time

    this->received_probs.init(recvd_probs.size().rows());
    for (int loop_n = 0; loop_n < this->length_n; loop_n++) {
        this->received_probs(loop_n).init(num_of_elements);
        alpha = real(0.0);
        for (int loop_e = 0; loop_e < num_of_elements; loop_e++) {
            tmp_prob = recvd_probs(loop_n, loop_e);
            // Clipping HACK
            this->perform_clipping(tmp_prob);
            this->received_probs(loop_n)(loop_e) = tmp_prob;
            alpha += tmp_prob;
        }
        assertalways(alpha != real(0.0));
        this->received_probs(loop_n) /= alpha;
    }

    // this uses the description of the algorithm as given by
    // MacKay in Information Theory, Inference and Learning Algorithms(2003)
    // on page 560 - chapter 47.3

    // some helper variables
    int non_zeros = 0;
    int pos_n;

    // simply set q_mxn(0)=P_n(0)=P(x_n=0) and q_mxn(1)=P_n(1)=P(x_n=1)
    for (int loop_m = 0; loop_m < this->dim_m; loop_m++) {
        const array1i_t& N_m = this->pchk_matrix.get_row_idxs(loop_m);
        non_zeros = N_m.size().length();
        for (int loop_n = 0; loop_n < non_zeros; loop_n++) {
            pos_n = N_m(loop_n);
            this->marginal_probs(loop_m, pos_n).q_mxn =
                this->received_probs(pos_n);
            this->marginal_probs(loop_m, pos_n).r_mxn.init(num_of_elements);
            this->marginal_probs(loop_m, pos_n).r_mxn = 0.0;
        }
    }

#if DEBUG >= 2
    libbase::trace << " Memory Usage:\n ";
    libbase::trace << this->marginal_probs.size() *
                          sizeof(sum_prod_alg_abstract<GF_q, real>::marginals) /
                          double(1 << 20)
                   << " MB" << std::endl;

    libbase::trace << std::endl
                   << "The marginal matrix is given by:" << std::endl;
    this->print_marginal_probs(libbase::trace);
#endif
}

template <class GF_q, class real>
void
sum_prod_alg_trad<GF_q, real>::compute_r_mn(int pos_m, int loop_n)
{
    const array1i_t& N_m = this->pchk_matrix.get_row_idxs(pos_m);
    const libbase::vector<GF_q>& N_m_vals =
        this->pchk_matrix.get_row_vals(pos_m);

    // the number of remaining symbols that can vary
    int num_of_var_syms = N_m.size().length() - 1;
    int num_of_elements = GF_q::elements();
    // for each check node we need to consider
    // num_of_elements^num_of_var_symbols cases
    int num_of_cases = int(pow(num_of_elements, num_of_var_syms));
    int pos_n = N_m(loop_n);
    int bitmask = num_of_elements - 1;

    // only use the entries that are variable
    array1i_t rel_N_m;
    rel_N_m.init(num_of_var_syms);
    int indx = 0;
    for (int loop = 0; loop < num_of_var_syms; loop++) {
        if (indx == loop_n) {
            indx++;
        }
        rel_N_m(loop) = N_m(indx);
        indx++;
    }
    // go through all cases - this will use bitwise manipulation
    GF_q syndrome_sym = GF_q(0);
    GF_q h_m_n_dash;
    // GF_q h_m_n;
    GF_q tmp_chk_val;

    int int_sym_val;
    int bits;
    int pos_n_dash;
    real q_nm_prod = real(1.0);

    this->marginal_probs(pos_m, pos_n).r_mxn = 0.0;
    GF_q check_value = N_m_vals(loop_n);

    for (int loop1 = 0; loop1 < num_of_cases; loop1++) {
        bits = loop1;
        syndrome_sym = GF_q(0);
        q_nm_prod = 1.0;
        for (int loop2 = 0; loop2 < num_of_var_syms; loop2++) {

            pos_n_dash = rel_N_m(loop2);

            // extract int value of the first symbol
            int_sym_val = bits & bitmask;
            // shift bits to the right by the dimension of the finite field
            bits = bits >> GF_q::dimension();

            // the parity check symbol at this position
            h_m_n_dash = N_m_vals(loop2);
            // compute the value that at this check
            tmp_chk_val = h_m_n_dash * GF_q(int_sym_val);

            // add it to the syndrome
            syndrome_sym = syndrome_sym + tmp_chk_val;
            // look up the prob that the chk_val was actually sent
            q_nm_prod *=
                this->marginal_probs(pos_m, pos_n_dash).q_mxn(int_sym_val);
        }
        // adjust the appropriate rmn value
        int_sym_val = syndrome_sym / check_value;
        this->marginal_probs(pos_m, pos_n).r_mxn(int_sym_val) += q_nm_prod;
    }
}

template <class GF_q, class real>
void
sum_prod_alg_trad<GF_q, real>::compute_q_mn(int loop_m, int pos_n)
{
    const array1i_t& M_n = this->pchk_matrix.get_col_idxs(pos_n);

    // initialise some helper variables
    int num_of_elements = GF_q::elements();
    array1d_t q_mn(this->received_probs(pos_n));

    int m_dash = 0;
    int pos_m = M_n(loop_m);

    // compute q_mn(sym) = a_mxn * P_n(sym) * \prod_{m'\in M(n)\m} r_m'xn(0) for
    // all sym in GF_q
    int size_of_M_n = M_n.size().length();
    for (int loop_m_dash = 0; loop_m_dash < size_of_M_n; loop_m_dash++) {
        if (loop_m_dash != loop_m) {
            m_dash = M_n(loop_m_dash);
            for (int loop_e = 0; loop_e < num_of_elements; loop_e++) {
                q_mn(loop_e) *=
                    this->marginal_probs(m_dash, pos_n).r_mxn(loop_e);
            }
        }
    }
    // normalise the q_mxn's so that q_mxn_0+q_mxn_1=1

    real a_nxm = q_mn.sum(); // sum up the values in q_mn
    assertalways(a_nxm != real(0));
    q_mn /= a_nxm; // normalise

    // store the values
    this->marginal_probs(pos_m, pos_n).q_mxn = q_mn;
}

} // namespace libcomm

#include "gf.h"
#include "logrealfast.h"
#include "mpreal.h"

namespace libcomm
{

// Explicit Realizations
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>

using libbase::logrealfast;
using libbase::mpreal;

// clang-format off
#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define REAL_TYPE_SEQ \
      (double)(float)(logrealfast)(mpreal)

/* Serialization string: ldpc<type,real>
 * where:
 *      type = gf2 | gf4 ...
 *      real = double | logrealfast | mpreal
 */
#define INSTANTIATE(r, args) \
      template class sum_prod_alg_trad<BOOST_PP_SEQ_ENUM(args)>;
// clang-format on

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (GF_TYPE_SEQ)(REAL_TYPE_SEQ))

} // namespace libcomm
