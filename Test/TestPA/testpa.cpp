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

// libbase::vector<bool> -> arma::Col<int> (0/1)
static arma::Col<int> to_arma_ivec(const libbase::vector<bool>& v) {
    const int n = v.size();
    arma::Col<int> out(static_cast<arma::uword>(n));
    for (int i = 0; i < n; ++i) out(i) = v(i) ? 1 : 0;  // note: v(i)
    return out;
}

// arma::Col<int> (0/1) -> libbase::vector<bool>
static libbase::vector<bool> arma_to_libbase_bvec(const arma::Col<int>& v) {
    libbase::vector<bool> out(static_cast<int>(v.n_elem));
    for (arma::uword i = 0; i < v.n_elem; ++i) out(static_cast<int>(i)) = (v(i) & 1) != 0;
    return out;
}

// pretty-print libbase::vector<bool> as contiguous 0/1
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
    const int L = 3;    // rows (hashed key length)
    const int N = 10;   // cols (original key length), typically k > L

    const int sv_len = L + N - 1;

    // RNG
    libbase::randgen rng;
    rng.seed(7);

    // 1) Starting vector as libbase::vector<bool>
    libbase::vector<bool> sv_bits(sv_len);
    for (int i = 0; i < sv_len; ++i)
        sv_bits(i) = (rng.ival(2) != 0);   // 0 or 1 with equal probability

    // 2) Convert to Armadillo ints
    arma::Col<int> starting_vector = to_arma_ivec(sv_bits);

    // 3) Generate Toeplitz(L×N): c = sv[L-1::-1], r = sv[L-1:]
    arma::Col<int> c = arma::reverse(starting_vector.head(static_cast<arma::uword>(L))); // first column
    arma::Col<int> r = starting_vector.tail(static_cast<arma::uword>(N));                // first row
    arma::Mat<int> T = arma::toeplitz(c, r);

    // Sanity Checks
    BOOST_TEST(static_cast<int>(T.n_rows) == L);
    BOOST_TEST(static_cast<int>(T.n_cols) == N);

    // Print the Toeplitz matrix
    std::cout << "Toeplitz matrix T (" << T.n_rows << " x " << T.n_cols << "):\n";
    T.print("T:");

    std::cout << "T size: " << (T.n_rows) << " x " << (T.n_cols) << "\n";

    // 4) Random libbase bit-vector 'key' of size N
    libbase::vector<bool> key_bits(N);
    for (int i = 0; i < N; ++i) key_bits(i) = (rng.ival(2) != 0);
    print_bvec(key_bits, "Generated vector s"); // libbase bool bits

    // Convert key to arma ints
    arma::Col<int> vector_s = to_arma_ivec(key_bits); // Bob's vector s

    // 5) Hashed key: y = (T * key) mod 2
    arma::Col<int> y = T * vector_s;
    in_place_mod2(y);

    std::cout << "Final Secret Key / Hashed key (arma, 0/1): ";
    for (arma::uword i = 0; i < y.n_elem; ++i) std::cout << y(i);
    std::cout << std::endl;

    // 6) Convert back to libbase::vector<bool>
    libbase::vector<bool> y_bits = arma_to_libbase_bvec(y);
    print_bvec(y_bits, "y Final Secret Key/(hashed bits) as a libbase bool vector");
}
