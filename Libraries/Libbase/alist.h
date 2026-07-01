/*!
 * \file
 *
 * Copyright (c) 2025 Mark Mizzi
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

#ifndef __alist_h
#define __alist_h

#include "matrix.h"
#include "vector.h"

#include <vector>

namespace libbase
{

template <typename GF_q>
class alist;

template <typename GF_q>
std::ostream& operator<<(std::ostream&, const alist<GF_q>&);

template <typename GF_q>
std::istream& operator>>(std::istream&, alist<GF_q>&);

/** \brief Sparse matrix representation largely implementing MacKay's alist
 * format (extended for non-binary codes)
 */
template <typename GF_q>
class alist
{
private:
    void test_invariant() const
    {
#ifndef NDEBUG
        assert(row_idxs.size() == row_vals.size());
        assert(col_idxs.size() == col_vals.size());

        int row_tanner_edges = 0;
        for (auto const& x : row_idxs) {
            row_tanner_edges += x.size().length();
            assert(x.size().length() > 0);
        }
        int col_tanner_edges = 0;
        for (auto const& x : col_idxs) {
            col_tanner_edges += x.size().length();
            assert(x.size().length() > 0);
        }
        assert(row_tanner_edges == col_tanner_edges);
#endif
    }

protected:
    /** \brief Non-zero indices for each row of the matrix.
     */
    std::vector<vector<int>> row_idxs;
    /** \brief Non-zero indices for each col of the matrix.
     */
    std::vector<vector<int>> col_idxs;
    /** \brief Non-zero values for each row of the matrix.
     */
    std::vector<vector<GF_q>> row_vals;
    /** \brief Non-zero values for each column of the matrix.
     */
    std::vector<vector<GF_q>> col_vals;

public:
    alist(std::vector<vector<int>>&& col_idxs,
          std::vector<vector<GF_q>>&& col_vals,
          int rows,
          const vector<int>& row_weights)
        : row_idxs(rows), col_idxs(col_idxs), row_vals(rows), col_vals(col_vals)
    {
        assertalways(col_idxs.size() == col_vals.size());

        // keep track of the last idx written in row_idxs(row) (and
        // row_vals(row)) for each row
        vector<int> row_last_idx;
        row_last_idx.init(rows);
        row_last_idx = 0;

        for (int row = 0; row < rows; row++) {
            row_idxs[row].init(row_weights(row));
            row_vals[row].init(row_weights(row));
        }

        for (int col = 0; col < cols(); col++) {
            for (int loop_r = 0; loop_r < col_idxs[col].size().length();
                 loop_r++) {
                int row = col_idxs[col](loop_r);
                int val = col_vals[col](loop_r);

                row_idxs[row](row_last_idx(row)) = col;
                row_vals[row](row_last_idx(row)) = val;

                row_last_idx(row)++;
            }
        }
    }

    alist() = default;
    ~alist() = default;

    alist(const alist&) = default;
    alist(alist&&) = default;

    alist& operator=(const alist&) = default;
    alist& operator=(alist&&) = default;

    /*! \name Conversion to/from equivalent dense reprs. */
private:
    //! \brief Common logic for constructor and \c operator= from \c
    //! libbase::matrix
    void from_matrix(const libbase::matrix<GF_q>&);

public:
    //! \brief copy from libbase::matrix
    alist(const matrix<GF_q>& x) { from_matrix(x); }
    alist& operator=(const matrix<GF_q>& x)
    {
        from_matrix(x);
        return *this;
    }
    //! \brief copy to standard matrix
    operator matrix<GF_q>() const;
    //! @}

    int cols() const { return col_idxs.size(); }
    int rows() const { return row_idxs.size(); }

    int max_col_weight() const
    {
        int max_num = 0;
        for (const auto& x : col_idxs) {
            if (max_num < x.size().length()) {
                max_num = x.size().length();
            }
        }
        return max_num;
    }
    int max_row_weight() const
    {
        int max_num = 0;
        for (const auto& x : row_idxs) {
            if (max_num < x.size().length()) {
                max_num = x.size().length();
            }
        }
        return max_num;
    }

    const vector<int>& get_row_idxs(int row) const { return row_idxs[row]; }
    const vector<int>& get_col_idxs(int col) const { return col_idxs[col]; }
    const vector<GF_q>& get_row_vals(int row) const { return row_vals[row]; }
    const vector<GF_q>& get_col_vals(int col) const { return col_vals[col]; }

    vector<int> col_weights() const
    {
        vector<int> weights;
        weights.init(cols());

        int i = 0;
        for (const auto& x : col_idxs) {
            weights(i) = x.size().length();
            i++;
        }

        return weights;
    }
    vector<int> row_weights() const
    {
        vector<int> weights;
        weights.init(rows());

        int i = 0;
        for (const auto& x : row_idxs) {
            weights(i) = x.size().length();
            i++;
        }

        return weights;
    }

    /*! \name Arithmetic operations. */
    vector<GF_q> operator*(const vector<GF_q>& x) const;
    //! @}

    /*! \name Linear Algebra. */
    //! \brief inplace matrix transpose
    void transpose();
    //! @}

    /*! \name Serialization. */
    /** \brief Serialize from MacKay alist format.
     *
     * The exact format (for binary codes only) is described
     * <a href="https://www.inference.org.uk/mackay/codes/alist.html">here</a>
     */
    friend std::istream& operator>> <>(std::istream&, alist&);
    /** \brief Serialize to MacKay alist format.
     *
     * The exact format (for binary codes only) is described
     * <a href="https://www.inference.org.uk/mackay/codes/alist.html">here</a>
     */
    friend std::ostream& operator<< <>(std::ostream&, const alist&);
    //! @}
};

template <typename GF_q>
std::istream&
operator>>(std::istream& sin, alist<GF_q>& a)
{
    assertalways(sin.good());
    int num_of_elements = GF_q::elements();
    bool nonbinary = (num_of_elements > 2);
    int cols, rows, max_col_weight, max_row_weight;

    sin >> libbase::eatcomments >> cols >> libbase::verify;
    sin >> libbase::eatcomments >> rows >> libbase::verify;
    if (nonbinary) {
        int q;
        sin >> libbase::eatcomments >> q >> libbase::verify;
        assertalways(num_of_elements == q);
    }

    sin >> libbase::eatcomments >> max_col_weight >> libbase::verify;
    sin >> libbase::eatcomments >> max_row_weight >> libbase::verify;

    // read the col weights and ensure they are sensible
    int tmp_col_weight;
    a.col_idxs = std::vector<vector<int>>(cols);
    a.col_vals = std::vector<vector<GF_q>>(cols);
    vector<int> col_weights;
    col_weights.init(cols);
    for (int loop1 = 0; loop1 < cols; loop1++) {
        sin >> libbase::eatcomments >> tmp_col_weight >> libbase::verify;
        // is it between 1 and max_col_weight?
        assertalways((1 <= tmp_col_weight) &&
                     (tmp_col_weight <= max_col_weight));
        a.col_idxs[loop1].init(tmp_col_weight);
        a.col_vals[loop1].init(tmp_col_weight);
        col_weights(loop1) = tmp_col_weight;
    }

    // read the row weights and ensure they are sensible
    int tmp_row_weight;
    a.row_idxs = std::vector<vector<int>>(rows);
    a.row_vals = std::vector<vector<GF_q>>(rows);
    vector<int> row_weights;
    row_weights.init(rows);
    for (int loop1 = 0; loop1 < rows; loop1++) {
        sin >> libbase::eatcomments >> tmp_row_weight >> libbase::verify;
        // is it between 1 and max_row_weight?
        assertalways((1 <= tmp_row_weight) &&
                     (tmp_row_weight <= max_row_weight));
        a.row_idxs[loop1].init(tmp_row_weight);
        a.row_vals[loop1].init(tmp_row_weight);
        row_weights(loop1) = tmp_row_weight;
    }

    // read the non-zero entries of the parity check matrix col by col
    // and ensure they make sense
    int tmp_entries;
    int tmp_pos;
    int tmp_val = 1; // this is the default value for the binary case
    for (int loop1 = 0; loop1 < cols; loop1++) {
        // read in the non-zero row entries
        tmp_entries = col_weights(loop1);
        a.col_idxs[loop1].init(tmp_entries);
        a.col_vals[loop1].init(tmp_entries);
        for (int loop2 = 0; loop2 < tmp_entries; loop2++) {
            sin >> libbase::eatcomments >> tmp_pos >> libbase::verify;
            tmp_pos--; // we start counting at 0 internally
            a.col_idxs[loop1](loop2) = tmp_pos;
            assertalways((0 <= tmp_pos) && (tmp_pos < rows));
            // read the non-zero element in the non-binary case
            if (nonbinary) {
                sin >> libbase::eatcomments >> tmp_val >> libbase::verify;
                assertalways((0 <= tmp_val) && (tmp_val < num_of_elements));
            }
            a.col_vals[loop1](loop2) = GF_q(tmp_val);
        }
        // discard any padded 0 zeros if necessary
        for (int loop2 = 0; loop2 < (max_col_weight - tmp_entries); loop2++) {
            sin >> libbase::eatcomments >> tmp_pos >> libbase::verify;
            assertalways(0 == tmp_pos);
            if (nonbinary) {
                sin >> libbase::eatcomments >> tmp_val >> libbase::verify;
                assertalways((0 == tmp_val));
            }
        }
    }

    // read the non-zero entries of the parity check matrix row by row
    for (int loop1 = 0; loop1 < rows; loop1++) {
        tmp_entries = row_weights(loop1);
        a.row_idxs[loop1].init(tmp_entries);
        a.row_vals[loop1].init(tmp_entries);
        for (int loop2 = 0; loop2 < tmp_entries; loop2++) {
            sin >> libbase::eatcomments >> tmp_pos >> libbase::verify;
            tmp_pos--; // we start counting at 0 internally
            a.row_idxs[loop1](loop2) = tmp_pos;
            assertalways((0 <= tmp_pos) && (tmp_pos < cols));
            // read the non-zero element in the non-binary case
            if (nonbinary) {
                sin >> libbase::eatcomments >> tmp_val >> libbase::verify;
                assertalways((0 <= tmp_val) && (tmp_val < num_of_elements));
            }
            a.row_vals[loop1](loop2) = GF_q(tmp_val);
            // TODO: Add check to ensure values in row_vals match up with values
            // in col_vals.
        }
        // discard any padded 0 zeros if necessary
        for (int loop2 = 0; loop2 < (max_row_weight - tmp_entries); loop2++) {
            sin >> libbase::eatcomments >> tmp_pos >> libbase::verify;
            assertalways(0 == tmp_pos);
            if (nonbinary) {
                sin >> libbase::eatcomments >> tmp_val >> libbase::verify;
                assertalways((0 == tmp_val));
            }
        }
    }

    a.test_invariant();
    return sin;
}

template <typename GF_q>
std::ostream&
operator<<(std::ostream& sout, const alist<GF_q>& a)
{
    assertalways(sout.good());
    a.test_invariant();

    int num_of_elements = GF_q::elements();
    bool nonbinary = (num_of_elements > 2);
    int max_col_weight = a.max_col_weight();
    int max_row_weight = a.max_row_weight();

    // alist format version
    sout << a.cols() << " " << a.rows();
    if (nonbinary) {
        sout << " " << num_of_elements;
    }
    sout << std::endl;
    sout << max_col_weight << " " << max_row_weight << std::endl;
    a.col_weights().serialize(sout, " ");
    a.row_weights().serialize(sout, " ");
    int num_of_non_zeros;
    int gf_val_int;
    int tmp_pos;

    // positions per column (and the non-zero values associated with them in the
    // non-binary case)
    for (int loop1 = 0; loop1 < a.cols(); loop1++) {
        num_of_non_zeros = a.col_idxs[loop1].size();
        for (int loop2 = 0; loop2 < num_of_non_zeros; loop2++) {
            tmp_pos = a.col_idxs[loop1](loop2);
            tmp_pos++;
            sout << tmp_pos << " ";
            if (nonbinary) {
                gf_val_int = a.col_vals[loop1](loop2);
                sout << gf_val_int << " ";
            }
        }
        // add 0 zeros if necessary
        for (int loop2 = 0; loop2 < (max_col_weight - num_of_non_zeros);
             loop2++) {
            sout << "0 ";
            if (nonbinary) {
                sout << "0 ";
            }
        }
        sout << std::endl;
    }

    // positions per row (and the non-zero values associated with them in the
    // non-binary case)
    for (int loop1 = 0; loop1 < a.rows(); loop1++) {
        num_of_non_zeros = a.row_idxs[loop1].size();
        for (int loop2 = 0; loop2 < num_of_non_zeros; loop2++) {
            tmp_pos = a.row_idxs[loop1](loop2);
            tmp_pos++;
            sout << tmp_pos << " ";
            if (nonbinary) {
                gf_val_int = a.row_vals[loop1](loop2);
                sout << gf_val_int << " ";
            }
        }
        // add 0 zeros if necessary
        for (int loop2 = 0; loop2 < (max_row_weight - num_of_non_zeros);
             loop2++) {
            sout << "0 ";
            if (nonbinary) {
                sout << "0 ";
            }
        }
        sout << std::endl;
    }

    return sout;
}

template <typename GF_q>
void
alist<GF_q>::from_matrix(const matrix<GF_q>& x)
{
    int rows = x.size().rows(), cols = x.size().cols();
    row_idxs.resize(rows);
    row_vals.resize(rows);
    col_idxs.resize(cols);
    col_vals.resize(cols);

    // populate row indexes and values
    for (int row = 0; row < rows; row++) {
        // compute weight for this row
        int row_weight = 0;
        for (int col = 0; col < cols; col++) {
            if (x(row, col) != GF_q(0)) {
                row_weight++;
            }
        }
        // initialize vectors with correct size
        row_idxs[row].init(row_weight);
        row_vals[row].init(row_weight);
        // populate idx and vals vectors.
        int loop1 = 0;
        for (int col = 0; col < cols; col++) {
            GF_q val = x(row, col);
            if (val != GF_q(0)) {
                row_idxs[row](loop1) = col;
                row_vals[row](loop1) = val;
                loop1++;
            }
        }
    }

    // populate col indexes and values
    for (int col = 0; col < cols; col++) {
        // compute weight for this col
        int col_weight = 0;
        for (int row = 0; row < rows; row++) {
            if (x(row, col) != GF_q(0)) {
                col_weight++;
            }
        }
        // initialize vectors with correct size
        col_idxs[col].init(col_weight);
        col_vals[col].init(col_weight);
        // populate idx and vals vectors.
        int loop1 = 0;
        for (int row = 0; row < rows; row++) {
            GF_q val = x(row, col);
            if (val != GF_q(0)) {
                col_idxs[col](loop1) = row;
                col_vals[col](loop1) = val;
                loop1++;
            }
        }
    }
}

template <typename GF_q>
alist<GF_q>::operator matrix<GF_q>() const
{
    matrix<GF_q> m;
    m.init(rows(), cols());

    for (int row = 0; row < rows(); row++) {
        for (int loop1 = 0; loop1 < row_idxs[row].size().length(); loop1++) {
            int col = row_idxs[row](loop1);
            m(row, col) = row_vals[row](loop1);
        }
    }

    return m;
}

template <typename GF_q>
vector<GF_q>
alist<GF_q>::operator*(const vector<GF_q>& x) const
{
    vector<GF_q> res;
    res.init(rows());

    for (int row = 0; row < rows(); row++) {
        res(row) = 0;
        for (int loop1 = 0; loop1 < row_idxs[row].size().length(); loop1++) {
            int col = row_idxs[row](loop1);
            GF_q val = row_vals[row](loop1);
            res(row) += x(col) * val;
        }
    }

    return res;
}

template <typename GF_q>
void
alist<GF_q>::transpose()
{
    std::swap(row_idxs, col_idxs);
    std::swap(row_vals, col_vals);
}

} // namespace libbase

#endif // __alist_h