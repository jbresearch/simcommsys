/*!
 * \file
 * \brief Boost unit tests to test the Toeplitz matrix generation.
 */

#define BOOST_TEST_MODULE testpa
#include <boost/test/included/unit_test.hpp>

#include <armadillo>
#include <iostream>
#include <sstream>

#include <vector.h>   // libbase::vector
#include <randgen.h>  // libbase::randgen

/* Notes for me:
1. arma::Col<T> is Armadillo’s type for a column vector (a matrix with 1 column) with element of Type T.
2. arma::uword is Armadillo’s internal type for array indices and sizes. It is defined as an unsigned integer type large enough to hold sizes and indices across platforms. It is equivalent to the unsigned int i.e. std::size_t in the STL library.
*/

/* Code Acknowledgement: The below C++ code is a reimplementation of the C++ Code done by Dr. Ing. Chris Galea as found in the GitLab library in the following GitLab repository under the modularise_hungary branch: https://dsrg-ict.research.um.edu.mt/qkd/ldpc-codes/-/tree/modularise_hungary_code/Other/Botond/Code/privacy_amplification?ref_type=heads */

// Converts libbase::vector<bool> to an arma::Col<int> (0/1)
static arma::Col<int> to_arma_ivec(const libbase::vector<bool>& v) {
    const int n = v.size();
    arma::Col<int> out(static_cast<arma::uword>(n));
    for (int i = 0; i < n; ++i) out(i) = v(i) ? 1 : 0;  // note: v(i)
    return out;
}

// Converts an arma::Col<int> (0/1) to a libbase::vector<bool>
static libbase::vector<bool> arma_to_libbase_bvec(const arma::Col<int>& v) {
    libbase::vector<bool> out(static_cast<int>(v.n_elem));
    for (arma::uword i = 0; i < v.n_elem; ++i) out(static_cast<int>(i)) = (v(i) & 1) != 0;
    return out;
}

// Prints a libbase::vector<bool> as contiguous 0/1
static void print_bvec(const libbase::vector<bool>& v, const char* label) {
    std::ostringstream oss;
    oss << label << " (len=" << v.size() << "): ";
    for (int i = 0; i < v.size(); ++i) oss << (v(i) ? '1' : '0');
    BOOST_TEST_MESSAGE(oss.str());
    std::cout << oss.str() << '\n';
}

// in-place mod-2 for arma ints
static void in_place_mod2(arma::Col<int>& v) {
    for (arma::uword i = 0; i < v.n_elem; ++i) v(i) &= 1;
}

BOOST_AUTO_TEST_CASE(generate_and_print_toeplitz_from_libbase_bits) {
    const int L = 3;    // rows (final hashed key length)
    const int N = 10;   // cols (original key length), vectors and s_hat with length k where k > L

    const int sv_len = L + N - 1; // Length of starting vector.

    // RNG
    libbase::randgen rng;
    rng.seed(7);

    // 1) Generate Starting vector as libbase::vector<bool>
    libbase::vector<bool> starting_vector_libbase (sv_len);
    for (int i = 0; i < sv_len; ++i)
        starting_vector_libbase(i) = (rng.ival(2) != 0);   // 0 or 1 with equal probability

    // 2) Convert to Armadillo ints
    arma::Col<int> starting_vector_arma = to_arma_ivec(starting_vector_libbase);

    // 3) Generate Toeplitz(L×N): c = sv[L-1::-1], r = sv[L-1:]
    arma::Col<int> c = arma::reverse(starting_vector_arma.head(static_cast<arma::uword>(L))); // first column
    arma::Col<int> r = starting_vector_arma.tail(static_cast<arma::uword>(N));                // first row
    arma::Mat<int> T = arma::toeplitz(c, r);

    // Sanity Checks
    BOOST_TEST(static_cast<int>(T.n_rows) == L);
    BOOST_TEST(static_cast<int>(T.n_cols) == N);

    // Print the Toeplitz matrix
    std::cout << "Toeplitz matrix T (" << T.n_rows << " x " << T.n_cols << "):\n";
    T.print("T:");

    std::cout << "T size: " << (T.n_rows) << " x " << (T.n_cols) << "\n";

    // 4) Generate random libbase bit-vector s of size N
    libbase::vector<bool> vector_s_libbase(N);
    for (int i = 0; i < N; ++i) vector_s_libbase(i) = (rng.ival(2) != 0);
    print_bvec(vector_s_libbase, "Generated vector s"); // libbase bool bits

    // Convert key to arma ints
    arma::Col<int> vector_s_arma = to_arma_ivec(vector_s_libbase); // Bob's vector s

    // 5) Hashed key: y = (T * key) mod 2
    arma::Col<int> y = T * vector_s_arma;

    // 6) Convert elements of y from ints to modulo-2.
    in_place_mod2(y);

    std::cout << "Final Secret Key / Hashed key (arma, 0/1): ";
    for (arma::uword i = 0; i < y.n_elem; ++i) std::cout << y(i);
    std::cout << std::endl;

    // 7) Convert back to libbase::vector<bool>
    libbase::vector<bool> y_bits = arma_to_libbase_bvec(y);
    print_bvec(y_bits, "y Final Secret Key/(hashed bits) as a libbase bool vector");
}
