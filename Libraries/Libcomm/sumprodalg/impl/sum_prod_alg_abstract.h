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

#ifndef SUM_PROD_ALG_ABSTRACT_H_
#define SUM_PROD_ALG_ABSTRACT_H_

#include "alist.h"
#include "config.h"
#include "hard_decision.h"
#include "matrix.h"
#include "random.h"
#include "sumprodalg/sum_prod_alg_inf.h"
#include "vector.h"
#include <limits>

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show set_syndrome
#ifndef NDEBUG
#    undef DEBUG
#    define DEBUG 1
#endif

namespace libcomm
{

/*! \brief Sum Product Algorithm(SPA) implementation
 *
 * Currently 2 types of the SPA: trad and gdl
 * The trad version computes the probabilities for the r__mxn's by computing
 * all the possible combinations of info symbols that satisfy the check node.
 * This can be very expensive computationally (especially when GF(q>2)
 * but it easy to code and understand.
 * The gdl version uses the fact that these probabilities can be grouped
 * differently using the distributive law and hence be computed much faster. The
 * version that is implemented here is based on Declercqs and Fossorier's 2006
 * paper: Decoding Algorithms for Nonbinary LDPC Codes over GF(q)
 */
template <class GF_q, class real = double>
class sum_prod_alg_abstract : public sum_prod_alg_inf<GF_q, real>
{
public:
    /*! \name Type definitions */
    typedef libbase::vector<real> array1d_t;
    typedef libbase::vector<int> array1i_t;
    typedef libbase::vector<array1i_t> array1vi_t;
    typedef libbase::vector<array1d_t> array1vd_t;
    typedef libbase::matrix<real> array2d_t;

    typedef sum_prod_alg_inf<GF_q, real> Base;
    // @}

    /*! \brief constructor
     * initialise the main variables
     */
    sum_prod_alg_abstract(const libbase::alist<GF_q>& pchk_matrix)
        : length_n(pchk_matrix.cols()), dim_m(pchk_matrix.rows()),
          pchk_matrix(pchk_matrix)
    {
        this->init_timer("t_spa_iteration");
        this->marginal_probs.init(dim_m, length_n);
    }
    /*! \brief default destructor
     *
     */
    virtual ~sum_prod_alg_abstract()
    {
        // nothing to do
    }

    /*! \brief initialise the Sum Product algorithm with the relevant
     * probabilities
     *
     */
    virtual void spa_init(const array2d_t& ptable) = 0;
    /*! \brief this returns the type of the Sum Product algorithm
     *
     */
    virtual std::string spa_type() = 0;
    int get_iters() override { return this->num_iters; }

    /*! \brief carry out one iteration of the SPA
     * This method will carry out the horizontal and vertical step
     * of the SPA and store the result of a hard decision on posteriors in the
     * received_word
     */
    void spa_iteration(libbase::vector<GF_q>& received_word) override;
    /*! \brief Perform entire decoding process.
     */
    void spa_iteration_seq(libbase::vector<GF_q>& received_word, int max_iters) override
    {
        for (; this->num_iters < max_iters; this->num_iters++) {
            this->spa_iteration(received_word);

            if (is_converged(received_word))
                break;
        }
    }

    /*! \brief Perform the desired clipping
     *
     */
    virtual void perform_clipping(real& num)
    {
        if (1 == this->clipping_method) {
            // use standard clipping
            if (num < this->almostzero) {
                num = this->almostzero;
            }
        } else {
            // use zero clipping
            if (num <= real(0.0)) {
                num = this->almostzero;
            }
        }
    }

    void set_syndrome(const libbase::vector<GF_q>& syndrome) override
    {
        assert(syndrome.size() == this->dim_m || syndrome.size() == 0);

        this->syndrome = syndrome;

#if DEBUG >= 2
        libbase::trace << "sum_prod_alg::set_syndrome: " << this->syndrome;
#endif
    }

    void seedfrom(libbase::random& r)
    {
        // Call base method first
        Base::seedfrom(r);
        // Seed hard-decision box
        hd_functor.seedfrom(r);
    }

protected:
    /*! \brief carries out the horizontal step of SPA
     * The r_mxn probabilities are computed
     */
    virtual void compute_r_mn(int pos_m, int loop_n) = 0;
    /*! \brief carried out the horizontal step of the SPA
     * the q_mxn probabilities are computed
     */
    virtual void compute_q_mn(int loop_m, int pos_n) = 0;

private:
    void print_marginal_probs(std::ostream& sout);
    void print_marginal_probs(int col, std::ostream& sout);
    void compute_probs(array1vd_t& ro);

protected:
    /*! \brief Computes syndrome of received_word, returns true if this is 0. */
    bool is_converged(libbase::vector<GF_q>& received_word)
    {
        int dim_pchk = this->pchk_matrix.rows();
        bool dec_success = true;
        int num_of_entries = 0;
        int pos_n = 0;

        GF_q tmp_val = GF_q(0);
        for (int pos_m = 0; pos_m < dim_pchk && dec_success; pos_m++) {
            tmp_val = GF_q(0);
            const array1i_t& N_m = this->pchk_matrix.get_row_idxs(pos_m);
            const libbase::vector<GF_q>& N_m_vals =
                this->pchk_matrix.get_row_vals(pos_m);
            num_of_entries = N_m.size();
            for (int loop_n = 0; loop_n < num_of_entries; loop_n++) {
                pos_n = N_m(loop_n);
                tmp_val += N_m_vals(loop_n) * received_word(pos_n);
            }
            // check if the syndrome matches the expected value at this position
            if (syndrome.size() > 0) {
                dec_success = tmp_val == syndrome(pos_m);
            } else {
                dec_success = tmp_val == GF_q(0);
            }
        }

        return dec_success;
    }

protected:
    /*! \name Data structures
     * LDPC specific datastructure used by the Sum-Product Algorithm
     */

    /* see MacKay's Information Theory, Inference and Learning Algs (2003,
     * ch 47.3,pp 559-561) for a proper definition of the following variables.
     */

    //! this struct holds the probabilities that check m is satisfied if symbol
    //! n of the received word is
    // fixed at symbols and the other symbols(<>n) have separable distributions
    // given by q_mxn
    struct marginals {
        array1d_t q_mxn;
        array1d_t qmn_conv; //! this holds the fast FFT transforms of the q_mxns
        array1d_t r_mxn;
    };
    // @}

    // the number of cols
    int length_n;
    // the number of rows
    int dim_m;

    array1vd_t received_probs;

    //! \brief the pchk matrix which we use throughout decoding.
    libbase::alist<GF_q> pchk_matrix;

    //! this matrix holds the r_mxn probabilities
    libbase::matrix<marginals> marginal_probs;

    //! Hard-decision box
    hard_decision<libbase::vector, real, GF_q> hd_functor;

    //! Number of iterations used to decode last codeword
    int num_iters = 0;

    /*! \name Variables used only when decoding is done through calls to
     * spa_iteration().
     */
    /*! \brief Stores received codeword when one is found using spa_iteration().
     */
    libbase::vector<GF_q> received_word;
    /*! \brief Indicates whether a previous call to spa_iteration() has already
     * successfully found a codeword.
     *
     * \todo At the moment subclasses have to set this to \c false in their
     * implementation of \c spa_init(). Not the most maintanable setup.
     */
    bool decode_success = false;

     //! The syndrome used by the SPA algorithm
    libbase::vector<GF_q> syndrome;
};

} // namespace libcomm

#endif /* SUM_PROD_ALG_ABSTRACT_H_ */
