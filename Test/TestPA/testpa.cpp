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
#include <toeplitz_standard.h>  //libbasee::toeplitz_standard

#include "qkd/privacy_amplification.h"
#include "qkd/privacy_amplification/pa_standard_toeplitz.h"

using namespace libcomm;
using namespace libbase;
;

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

// --- Helpers ---

// libbase::matrix<T> -> arma::Mat<int> (for comparison / printing)
template <class T>
static arma::Mat<int> to_arma_imat(const libbase::matrix<T>& M) {
    const int R = M.size().rows();
    const int C = M.size().cols();
    arma::Mat<int> out(static_cast<arma::uword>(R), static_cast<arma::uword>(C));
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            out(i, j) = static_cast<int>(M(i, j));
    return out;
}

// Multiply y = (A @ key) mod 2 (length(y) = rows(A)); A can be bool or int
template <class TMat, class TKey>
static libbase::vector<int> matvec_mod2_libbase(const libbase::matrix<TMat>& A,
                                                const libbase::vector<TKey>& key)
{
    const int L = A.size().rows();
    const int N = A.size().cols();
    BOOST_REQUIRE(key.size() == N);

    libbase::vector<int> y(L);
    for (int i = 0; i < L; ++i) {
        int parity = 0;
        for (int j = 0; j < N; ++j) {
            const int aij = static_cast<int>(A(i, j)) & 1;
            const int xj  = static_cast<int>(key(j)) & 1;
            parity ^= (aij & xj);
        }
        y(i) = parity;
    }
    return y;
}

// Multiply y = (A @ key) mod q (q>=2); A, key are ints, result is int vector of residues
template <class TMat>
static libbase::vector<int> matvec_modq_libbase(const libbase::matrix<TMat>& A,
                                                const libbase::vector<int>& key,
                                                int q)
{
    const int L = A.size().rows();
    const int N = A.size().cols();
    BOOST_REQUIRE(key.size() == N);
    BOOST_REQUIRE(q >= 2);

    libbase::vector<int> y(L);
    for (int i = 0; i < L; ++i) {
        int acc = 0;
        for (int j = 0; j < N; ++j) {
            int aij = static_cast<int>(A(i, j)) % q; if (aij < 0) aij += q;
            int xj  = key(j) % q;                    if (xj  < 0) xj  += q;
            acc += aij * xj;
        }
        int r = acc % q; if (r < 0) r += q;
        y(i) = r;
    }
    return y;
}

// libbase::vector<T> -> arma::Col<int> (casts elements to int)
template <class T>
static arma::Col<int> to_arma_ivec(const libbase::vector<T>& v) {
    const int n = v.size();
    arma::Col<int> out(static_cast<arma::uword>(n));
    for (int i = 0; i < n; ++i) {
        // bool -> 0/1; ints unchanged; other numeric types cast to int
        out(i) = static_cast<int>(v(i));
    }
    return out;
}

/* Start of Boost Tests */
BOOST_AUTO_TEST_CASE(generate_and_print_toeplitz_from_libbase_bits) {
    std::cout << "Boost Test 1: generate_and_print_toeplitz_from_libbase_bits \n";
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


BOOST_AUTO_TEST_CASE(generate_toeplitz_with_libbase_impl_and_compare)
{
    std::cout << "\nBoost Test 2: Comparing Implementations of the Generation of Standard Toeplitz Matrices:" << std::endl;

    const int L = 3;
    const int N = 10;
    const int sv_len = L + N - 1;

    // RNG
    libbase::randgen rng;
    rng.seed(7);

    // 1) Generate starting vector as libbase::vector<bool>
    libbase::vector<bool> starting_vector_libbase(sv_len);
    for (int i = 0; i < sv_len; ++i)
        starting_vector_libbase(i) = (rng.ival(2) != 0);

    // 2) Reference matrix via Armadillo (as in your first test)
    arma::Col<int> starting_vector_arma = to_arma_ivec(starting_vector_libbase);
    arma::Col<int> c = arma::reverse(starting_vector_arma.head(static_cast<arma::uword>(L)));
    arma::Col<int> r = starting_vector_arma.tail(static_cast<arma::uword>(N));
    arma::Mat<int> T_ref = arma::toeplitz(c, r);

    // 3) Toeplitz via libbase implementation (exact same slicing)
    //    Class renamed to toeplitz_standard with single template parameter T.
    libbase::matrix<bool> T_lib_bool = libbase::toeplitz_standard::build<bool>(starting_vector_libbase, L, N);

    // 4) Compare sizes and entries
    BOOST_TEST(static_cast<int>(T_ref.n_rows) == T_lib_bool.size().rows());
    BOOST_TEST(static_cast<int>(T_ref.n_cols) == T_lib_bool.size().cols());

    arma::Mat<int> T_lib = to_arma_imat(T_lib_bool); // 0/1 ints for comparison

    // Element-wise equality
    for (arma::uword i = 0; i < T_ref.n_rows; ++i) {
        for (arma::uword j = 0; j < T_ref.n_cols; ++j) {
            BOOST_TEST(T_ref(i, j) == T_lib(i, j));
        }
    }

    std::cout << "Toeplitz via Armadillo:\n" << T_ref << std::endl;
    std::cout << "Toeplitz via Libbase:\n" << T_lib_bool << std::endl;
}


// --------- TEST 1: GF(2) ---------
BOOST_AUTO_TEST_CASE(hash_gf2_libbase_matches_armadillo)
{
    std::cout << "\nBoost Test 3: hash_gf2_libbase_matches_armadillo\n";
    // Small deterministic example
    const int L = 8;                 // rows (final hash length)
    const int N = 12;                 // cols (key length)
    const int sv_len = L + N - 1;    // = 4

    // starting vector sv (bool) and key = 101 (bool)
    libbase::vector<bool> sv_bool(sv_len);
    sv_bool(0) = 1; sv_bool(1) = 0; sv_bool(2) = 1; sv_bool(3) = 1;  // [1,0,1,1]

    libbase::vector<bool> key_bool(N);
    key_bool(0) = 1; key_bool(1) = 0; key_bool(2) = 1;               // "101"

    // ---- Armadillo reference: T = toeplitz( sv[L-1::-1], sv[L-1:] ) ----
    arma::Col<int> sv_arma = to_arma_ivec(sv_bool);
    arma::Col<int> c = arma::reverse(sv_arma.head(static_cast<arma::uword>(L)));  // first col
    arma::Col<int> r = sv_arma.tail(static_cast<arma::uword>(N));                 // first row
    arma::Mat<int> T_ref = arma::toeplitz(c, r);

    arma::Col<int> key_ref = to_arma_ivec(key_bool);
    arma::Col<int> y_ref = T_ref * key_ref;   // L×1
    in_place_mod2(y_ref);

    // ---- libbase Toeplitz (int) + libbase matrix×matrix multiply + mod 2 ----
    // Convert sv/key to ints (0/1)
    libbase::vector<int> sv_int(sv_len);
    for (int i = 0; i < sv_len; ++i) sv_int(i) = sv_bool(i) ? 1 : 0;
    libbase::vector<int> key_int(N);
    for (int j = 0; j < N; ++j) key_int(j) = key_bool(j) ? 1 : 0;

    // Build T as matrix<int>
    libbase::matrix<int> T_lib = libbase::toeplitz_standard::build<int>(sv_int, L, N);

    // Turn key (vector) into N×1 matrix; multiply Y = T * K; extract col 0
    libbase::matrix<int> K;   // will become (N×1)
    K = key_int;              // matrix::operator=(vector) -> single column
    libbase::matrix<int> Y = T_lib * K;

    libbase::vector<int> y_lib;     // length L
    Y.extractcol(y_lib, 0);

    // Reduce modulo 2
    for (int i = 0; i < y_lib.size(); ++i) y_lib(i) &= 1;

    // ---- Compare element-wise ----
    BOOST_TEST(static_cast<int>(y_ref.n_elem) == y_lib.size());
    for (int i = 0; i < y_lib.size(); ++i) {
        BOOST_TEST(y_ref(static_cast<arma::uword>(i)) == y_lib(i));
    }

    // Optional prints (handy while developing)
    std::cout << "Final Hashed Key in GF(2) using Armadillo y_ref: ";
    for (arma::uword i = 0; i < y_ref.n_elem; ++i) std::cout << y_ref(i) << (i + 1 == y_ref.n_elem ? "" : " ");
    std::cout << '\n';
    std::cout<<"Final Hashed Key in GF(2) using libbase y_lib: " << y_lib;
}

// --------- TEST 2: GF(16)  ---------
BOOST_AUTO_TEST_CASE(hash_gf16_libbase_matches_armadillo)
{
    std::cout << "\nBoost Test 4: hash_gf16_libbase_matches_armadillo\n";
    const int L = 3;
    const int N = 10;
    const int q = 16;
    const int sv_len = L + N - 1;

    libbase::randgen rng;
    rng.seed(123);

    // sv in 0..15
    libbase::vector<int> sv_int(sv_len);
    for (int i = 0; i < sv_len; ++i) sv_int(i) = rng.ival(q);

    // key in 0..15
    libbase::vector<int> key_int(N);
    for (int j = 0; j < N; ++j) key_int(j) = rng.ival(q);

    // ---- Armadillo reference ----
    arma::Col<int> sv_arma = to_arma_ivec(sv_int);
    arma::Col<int> c = arma::reverse(sv_arma.head(static_cast<arma::uword>(L)));
    arma::Col<int> r = sv_arma.tail(static_cast<arma::uword>(N));
    arma::Mat<int> T_ref = arma::toeplitz(c, r);

    arma::Col<int> key_ref = to_arma_ivec(key_int);
    arma::Col<int> y_ref = T_ref * key_ref;  // L×1
    for (arma::uword i = 0; i < y_ref.n_elem; ++i) {
        int v = y_ref(i) % q; if (v < 0) v += q;
        y_ref(i) = v;
    }

    // ---- libbase Toeplitz + libbase matrix×matrix multiply + mod q ----
    libbase::matrix<int> T_lib = libbase::toeplitz_standard::build<int>(sv_int, L, N);

    libbase::matrix<int> K;  // N×1
    K = key_int;
    libbase::matrix<int> Y = T_lib * K;

    libbase::vector<int> y_lib;  // L
    Y.extractcol(y_lib, 0);

    for (int i = 0; i < y_lib.size(); ++i) {
        int v = y_lib(i) % q; if (v < 0) v += q;
        y_lib(i) = v;
    }

    // ---- Compare ----
    BOOST_TEST(static_cast<int>(y_ref.n_elem) == y_lib.size());
    for (int i = 0; i < y_lib.size(); ++i) {
        BOOST_TEST(y_ref(static_cast<arma::uword>(i)) == y_lib(i));
    }

    std::cout << "Final Hashed Key in GF(16) using Armadillo y_ref: ";
    for (arma::uword i = 0; i < y_ref.n_elem; ++i) std::cout << y_ref(i) << (i + 1 == y_ref.n_elem ? "" : " ");
    std::cout << '\n';
    std::cout<<"Final Hashed Key in GF(16) using libbase y_lib: " << y_lib;
}

BOOST_AUTO_TEST_CASE(testing_pa_standard_toeplitz_without_serialization)
{
    std::cout << "Boost Test 5: Testing Privacy Amplification using Toeplitz Matrix without Serialization directly from object" << std::endl;
    // Construct the object directly
    libcomm::pa_standard_toeplitz<bool> pa_system;
    const int L = 3;
    const int N = 10;
    const int q = 2;

    pa_system.init(L, N, q);

    randgen r;
    r.seed(2602);

    pa_system.seedfrom(r);

    int start_vector_len = pa_system.generate_starting_vector_length();
    std::cout << "\n Privacy Amplification System Description = " << pa_system.description() << std::endl;
    std::cout << "\nStarting Vector Length = " << start_vector_len << std::endl;

    libbase::vector<bool> starting_vector = pa_system.generate_starting_vector(start_vector_len, q);


    std::cout << "\nStarting Vector:" << std::endl;
    for (int i =0;  i<start_vector_len; ++i)
    {
        std::cout << starting_vector(i);
    }
    std::cout <<"\n";

    libbase::matrix<bool> standard_toeplitz_matrix = pa_system.generate_toeplitz_matrix(starting_vector);

    std::cout << "\n Generated Standard Toeplitz Matrix:" << standard_toeplitz_matrix  << std::endl;
}
