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

/*
 * ldpc.cpp
 *
 *  Created on: 9 Jul 2009
 *      Author: swesemeyer
 */

#include "ldpc.h"
#include "alist.h"
#include "linear_code_utils.h"
#include "randgen.h"
#include "sumprodalg/spa_factory.h"
#include <cmath>
#include <cstdlib>
#include <sstream>

namespace libcomm
{

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show intermediate decoding output
#ifndef NDEBUG
#    undef DEBUG
#    define DEBUG 1
#endif

template <class GF_q, class real>
ldpc<GF_q, real>::ldpc(libbase::alist<GF_q> paritycheck_mat,
                       const int num_of_iters)
{
    // initialise the provided values;
    this->pchk_matrix = paritycheck_mat;
    this->max_iter = num_of_iters;

    // compute some values from the parity check matrix
    this->length_n = paritycheck_mat.cols();
    this->dim_pchk = paritycheck_mat.rows();

    this->row_weight = paritycheck_mat.row_weights();
    this->col_weight = paritycheck_mat.col_weights();
    this->max_row_weight = paritycheck_mat.max_row_weight();
    this->max_col_weight = paritycheck_mat.max_col_weight();

    this->reduce_to_ref = false;
    this->rand_prov_values = "provided";
    // we are done and can call init now.
    this->init();

    // use sensible default values for the rest
    std::string spa_type = "gdl";
    this->spa_alg =
        libcomm::spa_factory<GF_q, real>::get_spa(spa_type, this->pchk_matrix);

    std::string clipping_type = "zero";
    real almost_zero = real(1E-100);
    this->spa_alg->set_clipping(clipping_type, almost_zero);
}

template <class GF_q, class real>
void
ldpc<GF_q, real>::init()
{
    // compute the generator matrix for the code

    // only place where we expand into a dense repr. for now
    libbase::matrix<GF_q> pchk_dense = this->pchk_matrix;

    libbase::matrix<GF_q> genmatrix_dense;
    libbase::linear_code_utils<GF_q>::compute_dual_code(
        pchk_dense, genmatrix_dense, this->perm_to_systematic);

    this->dim_k = genmatrix_dense.size().rows();
    this->info_symb_pos.init(this->dim_k);

    if (this->reduce_to_ref == false) {
        // as we define the LDPC code by its parity check matrix,H , the
        // generator matrix will be of the form (P|I) provided H was in
        // systematic form when reduced to REF if H wasn't then
        // perm_to_systematic contains the permutation that transformed H
        // into systematic form which gives us the information we need to
        // extract the positions of the info symbols in G. In fact the last
        // k values of perm_to_systematic are those positions.
        for (int loop = 0; loop < this->dim_k; loop++) {
            this->info_symb_pos(loop) =
                this->perm_to_systematic((this->length_n - this->dim_k) + loop);
        }
    } else {
        // we reduce the generator matrix to REF format in the hope that the
        // info symbols will be in the first k positions and that we'll
        // therefore have a systematic code
        genmatrix_dense.reduce_to_ref();
        // we now need to find the pivots
        int posy = 0;
        for (int loop = 0; loop < this->dim_k; loop++) {
            while (genmatrix_dense(loop, posy) == GF_q(0)) {
                posy++;
            }
            this->info_symb_pos(loop) = posy;
        }
    }
    // convert back to sparse repr.
    this->gen_matrix = genmatrix_dense;
    this->gen_matrix.transpose();
}

template <class GF_q, class real>
void
ldpc<GF_q, real>::do_init_decoder(const array1vdbl_t& ptable)
{

    this->current_iteration = 0;

#if DEBUG >= 2
    libbase::trace << std::endl
                   << "The first 5 received likelihoods are:" << std::endl;
    libbase::trace << ptable.extract(0, 5);
#endif
    int num_of_elements = GF_q::elements();
    this->received_probs.init(this->length_n, num_of_elements);

    // cast the values from double to real
    for (int loop_n = 0; loop_n < this->length_n; loop_n++) {
        for (int loop_e = 0; loop_e < num_of_elements; loop_e++) {
            this->received_probs(loop_n, loop_e) = real(ptable(loop_n)(loop_e));
        }
    }

#if DEBUG >= 2
    libbase::trace << std::endl
                   << "Currently, the most likely received word is:"
                   << std::endl;
    this->received_word_hd.serialize(libbase::trace, " ");
#endif

    this->spa_alg->spa_init(this->received_probs);
}

template <class GF_q, class real>
void
ldpc<GF_q, real>::do_encode(const libbase::vector<int>& source,
                            libbase::vector<int>& encoded)
{
    libbase::vector<GF_q> source_gf;
    source_gf.init(source.size().length());

    for (int loop1 = 0; loop1 < source.size().length(); loop1++) {
        source_gf(loop1) = GF_q(source(loop1));
    }

    encoded = this->gen_matrix * source_gf;

#if DEBUG >= 2
    this->received_word_hd = encoded;
    //  extract the info symbols from the codeword word and compare them to the
    //  original
    for (int loop_i = 0; loop_i < this->dim_k; loop_i++) {
        assertalways(source(loop_i) == encoded(this->info_symb_pos(loop_i)));
    }
#endif
#if DEBUG >= 2
    libbase::trace << "The encoded word is:" << std::endl;
    encoded.serialize(libbase::trace, " ");
    libbase::trace << std::endl;
#endif
}

template <class GF_q, class real>
std::string
ldpc<GF_q, real>::description() const
{
    std::ostringstream sout;
    sout << "LDPC(n=" << this->length_n << ", m=" << this->dim_pchk
         << ", k=" << this->dim_k << ", spa=" << this->spa_alg->spa_type()
         << ", iter=" << this->max_iter
         << ", clipping=" << this->spa_alg->get_clipping_type()
         << ", almostzero=" << this->spa_alg->get_almostzero() << ")";

    const auto degenerate_rows = dim_pchk - (length_n - dim_k);

    if (degenerate_rows > 0) {
        sout << " [Code has " << degenerate_rows << " degenerate rows]";
    }

#if DEBUG >= 2
    this->serialize(libbase::trace);
    libbase::trace << std::endl;
#endif

#if DEBUG >= 2
    libbase::trace << "Its parity check matrix is given by:" << std::endl;
    libbase::trace << this->pchk_matrix << std::endl;

    libbase::trace << "Its generator matrix is given by:" << std::endl;
    libbase::trace << this->gen_matrix << std::endl;
    libbase::trace << "The information symbols are located in columns:"
                   << std::endl;
    for (int loop = 0; loop < this->dim_k; loop++) {
        libbase::trace << this->info_symb_pos(loop) + 1 << " ";
    }
    libbase::trace << std::endl;
#endif
    return sout.str();
}

/* object serialization - writing
 *
 * As an example, given the matrix (5,3) over GF(4):
 *
 * 2 1 0 0 1
 * 0 2 2 3 3
 * 3 0 2 1 0
 *
 * An example file would be
 * ldpc<gf2,double>
 * #version
 * 4
 * #SPA
 * trad
 * #iter
 * 10
 * #clipping method and almost_zero value
 * zero
 * 1e-100
 * # reduce generator matrix to REF? (true|false)
 * false
 * # length dim
 * 5 3
 * # max col/row weight
 * 2 2
 * #non-zero vals
 * provided
 * # col weights
 * 5
 * 2 2 2 2 2
 * # row weights
 * 3
 * 3 4 3
 * #non-zero pos in cols
 * 2
 * 1 3
 * 2
 * 1 2
 * 2
 * 2 3
 * 2
 * 2 3
 * 2
 * 1 2
 * #non-zero vals in cols
 * 2
 * 2 3
 * 2
 * 1 2
 * 2
 * 2 2
 * 2
 * 3 1
 * 2
 * 1 3
 *
 */
template <class GF_q, class real>
std::ostream&
ldpc<GF_q, real>::serialize(std::ostream& sout) const
{
    assertalways(sout.good());
    sout << "# Version" << std::endl;
    sout << 6 << std::endl;
    sout << "# SPA type (trad|gdl)" << std::endl;
    sout << this->spa_alg->spa_type() << std::endl;
    sout << "# Number of iterations" << std::endl;
    sout << this->max_iter << std::endl;
    sout << "# Clipping method (zero=replace only zeros, clip=replace values "
            "below almostzero)"
         << std::endl;
    sout << this->spa_alg->get_clipping_type() << std::endl;
    sout << "# Value of almostzero" << std::endl;
    sout << this->spa_alg->get_almostzero() << std::endl;
    sout << "# Length (n)" << std::endl;
    sout << this->length_n << std::endl;
    sout << "# Dimension (m)" << std::endl;
    sout << this->dim_pchk << std::endl;
    sout << "# Information symbols (k)" << std::endl;
    sout << this->dim_k << std::endl;
    sout << "# Pchk matrix max column weight" << std::endl;
    sout << this->max_col_weight << std::endl;
    sout << "# Pchk matrix max row weight" << std::endl;
    sout << this->max_row_weight << std::endl;

    sout << "# Pchk matrix column weight vector" << std::endl;
    sout << this->col_weight;
    sout << "# Pchk matrix row weight vector" << std::endl;
    sout << this->row_weight;

    sout << "# Pchk matrix non zero positions per col" << std::endl;
    for (int loop1 = 0; loop1 < this->length_n; loop1++) {
        sout << this->pchk_matrix.get_col_idxs(loop1) +
                    1; // we start counting from zero
    }

    // (always) output pchk matrix non-zero entries
    libbase::vector<int> non_zero_vals_in_col;
    sout << "# Pchk matrix non zero values per col" << std::endl;
    for (int loop1 = 0; loop1 < this->length_n; loop1++) {
        int num_of_non_zeros = this->pchk_matrix.get_col_idxs(loop1).size();
        non_zero_vals_in_col.init(num_of_non_zeros);
        for (int loop2 = 0; loop2 < num_of_non_zeros; loop2++) {
            int gf_val_int = this->pchk_matrix.get_col_vals(loop1)(loop2);
            assert(gf_val_int != GF_q(0));
            non_zero_vals_in_col(loop2) = gf_val_int;
        }
        sout << non_zero_vals_in_col;
    }

    sout << "# Generator matrix max column weight" << std::endl;
    sout << this->gen_matrix.max_col_weight() << std::endl;
    sout << "# Generator matrix max row weight" << std::endl;
    sout << this->gen_matrix.max_row_weight() << std::endl;

    sout << "# Generator matrix column weight vector" << std::endl;
    sout << this->gen_matrix.col_weights();
    sout << "# Generator matrix row weight vector" << std::endl;
    sout << this->gen_matrix.row_weights();

    sout << "# Generator matrix non zero positions per col" << std::endl;
    for (int loop1 = 0; loop1 < this->dim_k; loop1++) {
        sout << this->gen_matrix.get_col_idxs(loop1) +
                    1; // we start counting from zero
    }

    // (always) output generator matrix non-zero entries
    sout << "# Generator matrix non zero values per col" << std::endl;
    for (int loop1 = 0; loop1 < this->dim_k; loop1++) {
        int num_of_non_zeros = this->gen_matrix.get_col_idxs(loop1).size();
        non_zero_vals_in_col.init(num_of_non_zeros);
        for (int loop2 = 0; loop2 < num_of_non_zeros; loop2++) {
            int gf_val_int = this->gen_matrix.get_col_vals(loop1)(loop2);
            assert(gf_val_int != GF_q(0));
            non_zero_vals_in_col(loop2) = gf_val_int;
        }
        sout << non_zero_vals_in_col;
    }

    sout << "# Positions of information symbols in a codeword" << std::endl;
    sout << info_symb_pos;

    sout << "# Permutation required to make pchk matrix systematic"
         << std::endl;
    sout << perm_to_systematic;

    return sout;
}

/* object serialization - loading
 *
 * For an example, see the writing method
 */

template <class GF_q, class real>
std::istream&
ldpc<GF_q, real>::serialize(std::istream& sin)
{
    assertalways(sin.good());
    int version;
    sin >> libbase::eatcomments >> version >> libbase::verify;
    assertalways(version >= 2);

    std::string spa_type;
    sin >> libbase::eatcomments >> spa_type >> libbase::verify;
    sin >> libbase::eatcomments >> this->max_iter >> libbase::verify;
    assertalways(this->max_iter >= 1);
    // Default clipping settings for files with versions less than 3
    std::string clipping_type = "zero";
    real almost_zero = real(1E-100);
    if (version >= 3) {
        /* My method of avoiding probs of zero is labelled "zero", while
         * the method of clipping all probs below a certain value is "clip".
         * In either case we need to replace a value by almostzero.
         */
        sin >> libbase::eatcomments >> clipping_type >> libbase::verify;
        assertalways(("clip" == clipping_type) || ("zero" == clipping_type));
        double tmp_az;
        sin >> libbase::eatcomments >> tmp_az >> libbase::verify;
        almost_zero = real(tmp_az);
    }
    // Default flag for files with versions less than 4 and greater than 5
    this->reduce_to_ref = false;
    if (version == 5) {
        sin >> libbase::eatcomments >> this->reduce_to_ref >> libbase::verify;
    } else if (version == 4) {
        std::string tmp_flag;
        sin >> libbase::eatcomments >> tmp_flag >> libbase::verify;
        assertalways(("true" == tmp_flag) || ("false" == tmp_flag));
        if ("true" == tmp_flag) {
            this->reduce_to_ref = true;
        }
    }
    sin >> libbase::eatcomments >> this->length_n >> libbase::verify;
    sin >> libbase::eatcomments >> this->dim_pchk >> libbase::verify;
    if (version >= 6) {
        sin >> libbase::eatcomments >> this->dim_k >> libbase::verify;
    }

    sin >> libbase::eatcomments >> this->max_col_weight >> libbase::verify;
    sin >> libbase::eatcomments >> this->max_row_weight >> libbase::verify;

    libbase::randgen rng;
    // default for files with version >= 6
    this->rand_prov_values = "provided";
    if (version < 6) {
        // for versions < 6, user can specify how nz values are obtained.

        // are the non-zero values provided or do we randomly generate them?
        sin >> libbase::eatcomments >> this->rand_prov_values >>
            libbase::verify;
        assertalways(("ones" == this->rand_prov_values) ||
                     ("random" == this->rand_prov_values) ||
                     ("provided" == this->rand_prov_values));
        if ("random" == this->rand_prov_values) {
            // read the seed value;
            sin >> libbase::eatcomments >> this->seed >> libbase::verify;
            assertalways(this->seed >= 0);
            rng.seed(this->seed);
        }
    }
    // read the col weights and ensure they are sensible
    this->col_weight.init(this->length_n);
    sin >> libbase::eatcomments >> this->col_weight >> libbase::verify;
    assertalways((1 <= this->col_weight.min()) &&
                 (this->col_weight.max() <= this->max_col_weight));

    // read the row weights and ensure they are sensible
    this->row_weight.init(this->dim_pchk);
    sin >> libbase::eatcomments >> this->row_weight >> libbase::verify;
    assertalways((0 < this->row_weight.min()) &&
                 (this->row_weight.max() <= this->max_row_weight));

    std::vector<libbase::vector<int>> col_idxs(this->length_n);
    // read the non-zero entries pos per col
    for (int loop1 = 0; loop1 < this->length_n; loop1++) {
        col_idxs[loop1].init(this->col_weight(loop1));
        sin >> libbase::eatcomments >> col_idxs[loop1] >> libbase::verify;
        col_idxs[loop1] -= 1; // we start counting from zero.
        // ensure that the number of non-zero pos matches the previously read
        // value
        assertalways(col_idxs[loop1].size().length() ==
                     this->col_weight(loop1));
    }

    std::vector<libbase::vector<GF_q>> col_vals(this->length_n);
    // read in the non-zero entries per column
    const int num_of_non_zero_elements = GF_q::elements() - 1;
    for (int loop1 = 0; loop1 < this->length_n; loop1++) {
        const int tmp_entries = this->col_weight(loop1);
        col_vals[loop1].init(tmp_entries);
        if ("ones" == this->rand_prov_values) {
            // in the binary case the non-zero values are 1
            col_vals[loop1] = GF_q(1);
        } else if ("random" == this->rand_prov_values) {
            for (int loop2 = 0; loop2 < tmp_entries; loop2++) {
                col_vals[loop1](loop2) =
                    GF_q(1 + int(rng.ival(num_of_non_zero_elements)));
            }
            assertalways(col_vals[loop1].min() != GF_q(0));
        } else {
            sin >> libbase::eatcomments >> col_vals[loop1] >> libbase::verify;
            assertalways(col_vals[loop1].min() != GF_q(0));
        }
    }
    this->pchk_matrix = libbase::alist<GF_q>(std::move(col_idxs),
                                             std::move(col_vals),
                                             this->dim_pchk,
                                             this->row_weight);

    if (version < 6) {
        // for versions < 6, we have to call init() to populate the generator
        // matrix, perm_to_systematic and info_symb_pos fields.
        this->init();
    } else {
        // if version >= 6, we read generator matrix, perm_to_systematic and
        // info_symb_pos from file.
        int gen_matrix_max_col_weight, gen_matrix_max_row_weight;

        sin >> libbase::eatcomments >> gen_matrix_max_col_weight >>
            libbase::verify;
        sin >> libbase::eatcomments >> gen_matrix_max_row_weight >>
            libbase::verify;

        // read the col weights for gen_matrix and ensure they are sensible
        libbase::vector<int> gen_matrix_col_weights;
        gen_matrix_col_weights.init(this->dim_k);
        sin >> libbase::eatcomments >> gen_matrix_col_weights >>
            libbase::verify;
        assertalways(
            (1 <= gen_matrix_col_weights.min()) &&
            (gen_matrix_col_weights.max() <= gen_matrix_max_col_weight));

        // read the row weights and ensure they are sensible
        libbase::vector<int> gen_matrix_row_weights;
        gen_matrix_row_weights.init(this->length_n);
        sin >> libbase::eatcomments >> gen_matrix_row_weights >>
            libbase::verify;
        assertalways(
            (0 < gen_matrix_row_weights.min()) &&
            (gen_matrix_row_weights.max() <= gen_matrix_max_row_weight));

        std::vector<libbase::vector<int>> gen_matrix_col_idxs(this->dim_k);
        // read the non-zero entries pos per col
        for (int loop1 = 0; loop1 < this->dim_k; loop1++) {
            gen_matrix_col_idxs[loop1].init(gen_matrix_col_weights(loop1));
            sin >> libbase::eatcomments >> gen_matrix_col_idxs[loop1] >>
                libbase::verify;
            gen_matrix_col_idxs[loop1] -= 1; // we start counting from zero.
            // ensure that the number of non-zero pos matches the previously
            // read value
            assertalways(gen_matrix_col_idxs[loop1].size().length() ==
                         gen_matrix_col_weights(loop1));
        }

        std::vector<libbase::vector<GF_q>> gen_matrix_col_vals(this->dim_k);
        // read in the non-zero entries per column
        for (int loop1 = 0; loop1 < this->dim_k; loop1++) {
            gen_matrix_col_vals[loop1].init(gen_matrix_col_weights(loop1));
            sin >> libbase::eatcomments >> gen_matrix_col_vals[loop1] >>
                libbase::verify;
            assertalways(gen_matrix_col_vals[loop1].min() != GF_q(0));
        }

        // initialize generator matrix from data obtained from file.
        this->gen_matrix = libbase::alist<GF_q>(std::move(gen_matrix_col_idxs),
                                                std::move(gen_matrix_col_vals),
                                                this->length_n,
                                                gen_matrix_row_weights);

        // initialize info_symb_pos
        sin >> libbase::eatcomments >> this->info_symb_pos >> libbase::verify;

        // initialize perm_to_systematic
        sin >> libbase::eatcomments >> this->perm_to_systematic >>
            libbase::verify;
    }
    this->spa_alg =
        libcomm::spa_factory<GF_q, real>::get_spa(spa_type, this->pchk_matrix);
    this->spa_alg->set_clipping(clipping_type, almost_zero);
    return sin;
}

/*
 * This method outputs the alist format of this code as
 * described by MacKay @
 * http://www.inference.phy.cam.ac.uk/mackay/codes/alist.html
 *
 * n m q
 * max_n max_m
 * list of the number of non-zero entries for each column
 * list of the number of non-zero entries for each row
 * pos of each non-zero entry per col followed by their values (for GF(p>2))
 * pos of each non-zero entry per row followed by their values (for GF(p>2))
 *
 * where
 * - n is the length of the code
 * - m is the dimension of the parity check matrix
 * - q is only provided in non-binary cases where q=|GF(q)|
 * - max_n is the maxiumum number of non-zero entries per column
 * - max_m is the maximum number of non-zero entries per row
 * Note that the last set of positions and values are only used to
 * verify the information provided by the first set.
 *
 * Note that in the binary case the values are left out as they will be 1
 * anyway. An example row with 4 non-zero entries would look like this (assuming
 * gf<3,0xB> 1 1 3 3 9 7 10 2 ie the non-zero entries at pos 1,3,9 and 10 are
 * 1,3,7 and 2 respectively
 *
 * Similarly a column with 3 non-zero entries over gf<3,0xB> would look like:
 * 3 4 6 3 12 6
 * ie the non-zero entries at pos 3,6 and 12 are 4, 3 and 6 respectively
 *
 * in the binary case the above row and column would simply be given by
 * 3 6 12
 * 1 3 9 10
 *
 * Also note that the alist format expects cols/rows with weight less than
 * the max col/row weight to be padded with extra 0s, eg
 * if a code has max col weight of 5 and a given col only has weight 3 then
 * it would look like
 * 1 4 5 0 0 (each entry immediately followed by the non-zero values in case of
 * non-binary code) These additional 0s need to be ignored
 *
 */

template <class GF_q, class real>
std::ostream&
ldpc<GF_q, real>::write_alist(std::ostream& sout) const
{
    assertalways(sout.good());
    return sout << this->pchk_matrix;
}

/* loading of the  alist format of an LDPC code
 * This method expects the following format
 *
 * n m q
 * max_n max_m
 * list of the number of non-zero entries for each column
 * list of the number of non-zero entries for each row
 * pos of each non-zero entry per col followed by their values
 * pos of each non-zero entry per row followed by their values
 *
 * where
 * - n is the length of the code
 * - m is the dimension of the parity check matrix
 * - q is only provided in non-binary cases where q=|GF(q)|
 * - max_n is the maxiumum number of non-zero entries per column
 * - max_m is the maximum number of non-zero entries per row
 * Note that the last set of positions and values are only used to
 * verify the information provided by the first set.
 *
 * Note that in the binary case the values are left out as they will be 1
 * anyway. An example row with 4 non-zero entries would look like this (assuming
 * gf<3,0xB> 1 1 3 3 9 7 10 2 ie the non-zero entries at pos 1,3,9 and 10 are
 * 1,3,7 and 2 respectively
 *
 * Similarly a column with 3 non-zero entries over gf<3,0xB> would look like:
 * 3 4 6 3 12 6
 * ie the non-zero entries at pos 3,6 and 12 are 4, 3 and 6 respectively
 *
 * in the binary case the above row and column would simply be given by
 * 3 6 12
 * 1 3 9 10
 *
 * Also note that the alist format expects cols/rows with weight less than
 * the max col/row weight to be padded with extra 0s, eg
 * if a code has max col weight of 5 and a given col only has weight 3 then
 * it would look like
 * 1 4 5 0 0 (each entry immediately followed by the non-zero values in case of
 * non-binary code) These additional 0s need to be ignored
 *
 */

template <class GF_q, class real>
std::istream&
ldpc<GF_q, real>::read_alist(std::istream& sin)
{
    assertalways(sin.good());

    sin >> this->pchk_matrix;

    // compute some values from the parity check matrix
    this->length_n = this->pchk_matrix.cols();
    this->dim_pchk = this->pchk_matrix.rows();
    this->row_weight = this->pchk_matrix.row_weights();
    this->col_weight = this->pchk_matrix.col_weights();
    this->max_row_weight = this->pchk_matrix.max_row_weight();
    this->max_col_weight = this->pchk_matrix.max_col_weight();

    // set some default values
    this->max_iter = 100;
    this->reduce_to_ref = false;
    if (GF_q::dimension() == 1) {
        this->rand_prov_values = "ones";
    } else {
        this->rand_prov_values = "provided";
    }
    this->init();
    this->spa_alg =
        libcomm::spa_factory<GF_q, real>::get_spa("gdl", this->pchk_matrix);
    this->spa_alg->set_clipping("zero", real(1e-100));
    return sin;
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
#include <boost/preprocessor/stringize.hpp>

using libbase::logrealfast;
using libbase::mpreal;
using libbase::serializer;

// clang-format off
#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#ifndef USE_CUDA
#define REAL_TYPE_SEQ \
   (double)(float)(mpreal)(logrealfast)
#else
#define REAL_TYPE_SEQ \
   (double)(float)
#endif

/* Serialization string: ldpc<type,real>
 * where:
 *      type = gf2 | gf4 ...
 *      real = double | logrealfast | mpreal
 */
#define INSTANTIATE(r, args) \
      template class ldpc<BOOST_PP_SEQ_ENUM(args)>; \
      template <> \
      const serializer ldpc<BOOST_PP_SEQ_ENUM(args)>::shelper( \
            "codec", \
            "ldpc<" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0,args)) "," \
            BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(1,args)) ">", \
            ldpc<BOOST_PP_SEQ_ENUM(args)>::create);
// clang-format on

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (GF_TYPE_SEQ)(REAL_TYPE_SEQ))

} // namespace libcomm
