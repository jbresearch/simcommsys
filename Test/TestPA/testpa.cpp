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
#include "gf.h"

using namespace libcomm;
using namespace libbase;

// Print Helper Functions
template <class GFVec>
void print_gf_vector_as_ints(const GFVec& v) {
    for (int i = 0; i < v.size(); ++i)
        std::cout << int(v(i));
    std::cout << "\n";
}

template <class GFMat>
void print_gf_matrix_as_ints(const GFMat& M, int nrows, int ncols) {
    for (int r = 0; r < nrows; ++r) {
        for (int c = 0; c < ncols; ++c) {
            std::cout << int(M(r, c)) << (c + 1 < ncols ? '\t' : '\n');
        }
    }
}

// ---- element printer: gf16 as 4-bit binary, e.g. 2 -> "0010"
inline void print_gf16_bin4(std::ostream& os, const libbase::gf16& a) {
    unsigned v = static_cast<unsigned>(a) & 0xF;   // 0..15
    os << ((v >> 3) & 1) << ((v >> 2) & 1) << ((v >> 1) & 1) << (v & 1);
}

// Optional: gf2 as single bit
inline void print_gf2_bit(std::ostream& os, const libbase::gf2& a) {
    os << (static_cast<int>(a) & 1);
}

// ---- vector/matrix printers that use the element printer and avoid sign-compare
template <class Vec, class ElemPrinter>
void print_vec_bin(const Vec& v, ElemPrinter ep, bool with_spaces = false) {
    const int sz = static_cast<int>(v.size());
    for (int i = 0; i < sz; ++i) {
        ep(std::cout, v(i));
        if (with_spaces && i + 1 < sz) std::cout << ' ';
    }
    std::cout << '\n';
    std::cout << std::endl;
}

template <class Mat, class ElemPrinter>
void print_mat_bin(const Mat& M, int nrows, int ncols, ElemPrinter ep) {
    for (int r = 0; r < nrows; ++r) {
        for (int c = 0; c < ncols; ++c) {
            ep(std::cout, M(r, c));
            std::cout << (c + 1 < ncols ? '\t' : '\n');
        }
    }
}


BOOST_AUTO_TEST_CASE(testing_pa_standard_toeplitz_without_serialization_q_2_inttype)
{
    std::cout << "*** Boost Test 1 ***: Testing Privacy Amplification using Standard Toeplitz Matrix without Serialization directly from object for q=2 with bool type " << std::endl;
    // Construct the object directly
    libcomm::pa_standard_toeplitz<bool> pa_system;
    const int L = 5;
    const int N = 15;
    const int q = 2;

    /*
    Test 3: Check that I get the same hashed key for the same inputted key

    Results from python script using L = 5, N = 15 and q = 2, bool type

    Key to be hashed = [0 1 1 1 0 0 0 0 0 0 0 1 1 1 0]

    (print from toeplitz_hashing.py) generated starting vector = [0 1 0 0 0 1 0 0 0 1 0 0 0 0 1 0 1 1 1]
    (print from base.py) Generated Standard Toeplitz Matrix =
    [[0 1 0 0 0 1 0 0 0 0 1 0 1 1 1]
    [0 0 1 0 0 0 1 0 0 0 0 1 0 1 1]
    [0 0 0 1 0 0 0 1 0 0 0 0 1 0 1]
    [1 0 0 0 1 0 0 0 1 0 0 0 0 1 0]
    [0 1 0 0 0 1 0 0 0 1 0 0 0 0 1]]
    Secure key length: 5
    Generated final secret (hased) key: [1 1 0 1 1]

    Result: Hashed keys matched.

    */
    pa_system.init(L, N, q);

    randgen r;
    r.seed(2602);

    pa_system.seedfrom(r);

    int starting_vector_len = pa_system.generate_starting_vector_length();
    std::cout << "\n Privacy Amplification System Description = " << pa_system.description() << std::endl;
    std::cout << "\nStarting Vector Length = " << starting_vector_len << std::endl;

    libbase::vector<bool> starting_vector(starting_vector_len); // Initialise starting vector with its length.

    // Test 3

        // --- Starting vector ---
    // Length = 19
    starting_vector(0)  = 0;
    starting_vector(1)  = 1;
    starting_vector(2)  = 0;
    starting_vector(3)  = 0;
    starting_vector(4)  = 0;
    starting_vector(5)  = 1;
    starting_vector(6)  = 0;
    starting_vector(7)  = 0;
    starting_vector(8)  = 0;
    starting_vector(9)  = 1;
    starting_vector(10) = 0;
    starting_vector(11) = 0;
    starting_vector(12) = 0;
    starting_vector(13) = 0;
    starting_vector(14) = 1;
    starting_vector(15) = 0;
    starting_vector(16) = 1;
    starting_vector(17) = 1;
    starting_vector(18) = 1;

    std::cout << "\nStarting Vector:" << std::endl;
    for (int i = 0; i < starting_vector.size(); ++i)
        std::cout << starting_vector(i);
    std::cout << "\n";

    libbase::matrix<bool> standard_toeplitz_matrix =
        pa_system.generate_toeplitz_matrix(starting_vector);

    std::cout << "\nGenerated Standard Toeplitz Matrix:"
            << standard_toeplitz_matrix << std::endl;

    // --- Pre-hashed key ---
    // Length = 15
    libbase::vector<bool> pre_hashed_key(15);
    pre_hashed_key(0)  = 0;
    pre_hashed_key(1)  = 1;
    pre_hashed_key(2)  = 1;
    pre_hashed_key(3)  = 1;
    pre_hashed_key(4)  = 0;
    pre_hashed_key(5)  = 0;
    pre_hashed_key(6)  = 0;
    pre_hashed_key(7)  = 0;
    pre_hashed_key(8)  = 0;
    pre_hashed_key(9)  = 0;
    pre_hashed_key(10) = 0;
    pre_hashed_key(11) = 1;
    pre_hashed_key(12) = 1;
    pre_hashed_key(13) = 1;
    pre_hashed_key(14) = 0;

    std::cout << "\nPre-hashed key:" << std::endl;
    for (int i = 0; i < pre_hashed_key.size(); ++i)
        std::cout << pre_hashed_key(i);
    std::cout << "\n";


    // --- Final secret key ---
    libbase::vector<bool> final_secret_key(L); // make sure L matches your test

    final_secret_key = pa_system.compute_hashed_key(
        standard_toeplitz_matrix,
        pre_hashed_key,
        L, N, q
    );

    std::cout << "The final secure key = " << final_secret_key << std::endl;
}


BOOST_AUTO_TEST_CASE(testing_pa_standard_toeplitz_without_serialization_q_16_inttype)
{
    std::cout << "*** Boost Test 2 ***: Testing Privacy Amplification using Toeplitz Matrix without Serialization directly from object for q=16 with int type " << std::endl;
    // Construct the object directly
    libcomm::pa_standard_toeplitz<int> pa_system;
    const int L = 5;
    const int N = 15;
    const int q = 16;

    pa_system.init(L, N, q);

    randgen r;
    r.seed(2602);

    pa_system.seedfrom(r);

    int starting_vector_len = pa_system.generate_starting_vector_length();
    std::cout << "\n Privacy Amplification System Description = " << pa_system.description() << std::endl;
    std::cout << "\nStarting Vector Length = " << starting_vector_len << std::endl;

    // libbase::vector<bool> starting_vector = pa_system.generate_starting_vector(start_vector_len, q);

    // libbase::vector<bool> starting_vector(starting_vector_len); // Initialise starting vector with its length.

    // Test 1 - Comparing to Python implementation with Chris (matrices matched) using starting vector = [0 1 0 0 0 1 0 0 0 1 0 0].
    // starting_vector(0) = 0;
    // starting_vector(1) = 1;
    // starting_vector(2) = 0;
    // starting_vector(3) = 0;
    // starting_vector(4) = 0;
    // starting_vector(5) = 1;
    // starting_vector(6) = 0;
    // starting_vector(7) = 0;
    // starting_vector(8) = 0;
    // starting_vector(9) = 1;
    // starting_vector(10) = 0;
    // starting_vector(11) = 0;

    // Test 2: Comparing to Python implementation with Chris using starting vector =  [0 1 0 0 0 1 0 0 0 1 0 0 0 0 1 0 1 1 1].

    // starting_vector(0) =  0;
    // starting_vector(1) =  1;
    // starting_vector(2) =  0;
    // starting_vector(3) =  0;
    // starting_vector(4) =  0;
    // starting_vector(5) =  1;
    // starting_vector(6) =  0;
    // starting_vector(7) =  0;
    // starting_vector(8) =  0;
    // starting_vector(9) =  1;
    // starting_vector(10) =  0;
    // starting_vector(11) =  0;
    // starting_vector(12) =  0;
    // starting_vector(13) =  0;
    // starting_vector(14) =  1;
    // starting_vector(15) =  0;
    // starting_vector(16) =  1;
    // starting_vector(17) =  1;
    // starting_vector(18) =  1;

    // std::cout << "\nStarting Vector from python:" << std::endl;
    // for (int i =0;  i<starting_vector_len; ++i)
    // {
    //     std::cout << starting_vector(i);
    // }
    // std::cout <<"\n";

    // libbase::matrix<bool> standard_toeplitz_matrix = pa_system.generate_toeplitz_matrix(starting_vector);

    // std::cout << "\n Generated Standard Toeplitz Matrix:" << standard_toeplitz_matrix  << std::endl;

    // libbase::vector<bool> pre_hashed_key(N); // initialises pre-hashed-key
    // pre_hashed_key(0) = 1;
    // pre_hashed_key(1) = 0;
    // pre_hashed_key(2) = 1;
    // pre_hashed_key(3) = 1;
    // pre_hashed_key(4) = 1;
    // pre_hashed_key(5) = 0;
    // pre_hashed_key(6) = 1;
    // pre_hashed_key(7) = 0;
    // pre_hashed_key(8) = 0;
    // pre_hashed_key(9) = 0;
    // pre_hashed_key(10) = 1;
    // pre_hashed_key(11) = 0;
    // pre_hashed_key(12) = 0;
    // pre_hashed_key(13) = 0;
    // pre_hashed_key(14) = 0;

    // libbase::vector<bool> final_secret_key(L); // initialises final secret key

    // final_secret_key = pa_system.compute_hashed_key(standard_toeplitz_matrix, pre_hashed_key, L, N, q); // computes the hashed key
    // std::cout << "The final secure key = " << final_secret_key << std::endl;

    // Test 3

    //     // --- Starting vector ---
    // // Length = 19
    // libbase::vector<bool> starting_vector(19);
    // starting_vector(0)  = 0;
    // starting_vector(1)  = 1;
    // starting_vector(2)  = 0;
    // starting_vector(3)  = 0;
    // starting_vector(4)  = 0;
    // starting_vector(5)  = 1;
    // starting_vector(6)  = 0;
    // starting_vector(7)  = 0;
    // starting_vector(8)  = 0;
    // starting_vector(9)  = 1;
    // starting_vector(10) = 0;
    // starting_vector(11) = 0;
    // starting_vector(12) = 0;
    // starting_vector(13) = 0;
    // starting_vector(14) = 1;
    // starting_vector(15) = 0;
    // starting_vector(16) = 1;
    // starting_vector(17) = 1;
    // starting_vector(18) = 1;

    // std::cout << "\nStarting Vector:" << std::endl;
    // for (int i = 0; i < starting_vector.size(); ++i)
    //     std::cout << starting_vector(i);
    // std::cout << "\n";

    // libbase::matrix<bool> standard_toeplitz_matrix =
    //     pa_system.generate_toeplitz_matrix(starting_vector);

    // std::cout << "\nGenerated Standard Toeplitz Matrix:"
    //         << standard_toeplitz_matrix << std::endl;

    // // --- Pre-hashed key ---
    // // Length = 15
    // libbase::vector<bool> pre_hashed_key(15);
    // pre_hashed_key(0)  = 0;
    // pre_hashed_key(1)  = 1;
    // pre_hashed_key(2)  = 1;
    // pre_hashed_key(3)  = 1;
    // pre_hashed_key(4)  = 0;
    // pre_hashed_key(5)  = 0;
    // pre_hashed_key(6)  = 0;
    // pre_hashed_key(7)  = 0;
    // pre_hashed_key(8)  = 0;
    // pre_hashed_key(9)  = 0;
    // pre_hashed_key(10) = 0;
    // pre_hashed_key(11) = 1;
    // pre_hashed_key(12) = 1;
    // pre_hashed_key(13) = 1;
    // pre_hashed_key(14) = 0;

    // // --- Final secret key ---
    // libbase::vector<bool> final_secret_key(L); // make sure L matches your test

    // final_secret_key = pa_system.compute_hashed_key(
    //     standard_toeplitz_matrix,
    //     pre_hashed_key,
    //     L, N, q
    // );

    // std::cout << "The final secure key = " << final_secret_key << std::endl;


    /*
    Test 1: Generated Standard Toeplitz Matrix from the Python script generated using the following:
     Matched with the implementation of Chris.

    Comparing to Python script of Chris. with L = 3, N = 10, q = 2, bool type.
    generated starting vector = [0 1 0 0 0 1 0 0 0 1 0 0]

    Generated Standard Toeplitz Matrix:
    [[0 0 0 1 0 0 0 1 0 0]
    [1 0 0 0 1 0 0 0 1 0]
    [0 1 0 0 0 1 0 0 0 1]]

    Result: Matrices matched.
    */

    /*
    Test 2: Generated Standard Toeplitz Matrix from the Python script generated using the following:

    Comparing to Python script of Chris. with L = 5, N = 15, q = 2 , bool type
    generated starting vector = [0 1 0 0 0 1 0 0 0 1 0 0 0 0 1 0 1 1 1].

    Generated Standard Toeplitz Matrix:
    [[0 1 0 0 0 1 0 0 0 0 1 0 1 1 1]
    [0 0 1 0 0 0 1 0 0 0 0 1 0 1 1]
    [0 0 0 1 0 0 0 1 0 0 0 0 1 0 1]
    [1 0 0 0 1 0 0 0 1 0 0 0 0 1 0]
    [0 1 0 0 0 1 0 0 0 1 0 0 0 0 1]]

    Result: Matrices matched.
    */

    /*

    Test 3: Check that I get the same hashed key for the same inputted key

    Results from python script using L = 5, N = 15 and q = 2, bool type

    Key to be hashed = [0 1 1 1 0 0 0 0 0 0 0 1 1 1 0]

    (print from toeplitz_hashing.py) generated starting vector = [0 1 0 0 0 1 0 0 0 1 0 0 0 0 1 0 1 1 1]
    (print from base.py) Generated Standard Toeplitz Matrix =
    [[0 1 0 0 0 1 0 0 0 0 1 0 1 1 1]
    [0 0 1 0 0 0 1 0 0 0 0 1 0 1 1]
    [0 0 0 1 0 0 0 1 0 0 0 0 1 0 1]
    [1 0 0 0 1 0 0 0 1 0 0 0 0 1 0]
    [0 1 0 0 0 1 0 0 0 1 0 0 0 0 1]]
    Secure key length: 5
    Generated final secret (hased) key: [1 1 0 1 1]

    Result: Hashed keys matched.

    Test 4 : Check that I get the same Toeplitz matrix and Hashed key for G16
    L = 5, N = 15, int type
    Key to be hashed = [ 2  5 10  7 15  5 14  6  5  5  2  0  9 11  0]
    (print from toeplitz_hashing.py) generated starting vector = [ 6  3 12 14 10  7 12  4  6  9  2  6 10 10  7  4  3  7  7]
    (print from base.py) Generated Standard Toeplitz Matrix = [[10  7 12  4  6  9  2  6 10 10  7  4  3  7  7]
    [14 10  7 12  4  6  9  2  6 10 10  7  4  3  7]
    [12 14 10  7 12  4  6  9  2  6 10 10  7  4  3]
    [ 3 12 14 10  7 12  4  6  9  2  6 10 10  7  4]
    [ 6  3 12 14 10  7 12  4  6  9  2  6 10 10  7]]
    Secure key length: 5
    Generated final secret (hased) key: [12  5 12 15  5]

    Results: Matched. Both implementations resulted in the same
    */

    // Test 4

    // --- Starting vector ---
    libbase::vector<int> starting_vector(L+N-1);
    int values[19] = {6, 3, 12, 14, 10, 7, 12, 4, 6, 9, 2, 6, 10, 10, 7, 4, 3, 7, 7};

    for (int i = 0; i < 19; i++) {
        starting_vector(i) = values[i];
    }

    std::cout << "\nStarting Vector:" << std::endl;
    for (int i = 0; i < starting_vector.size(); ++i)
        std::cout << starting_vector(i);
    std::cout << "\n";

    libbase::matrix<int> standard_toeplitz_matrix =
        pa_system.generate_toeplitz_matrix(starting_vector);

    std::cout << "\nGenerated Standard Toeplitz Matrix:"
            << standard_toeplitz_matrix << std::endl;

    // --- Pre-hashed key ---
    // Length = 15
    libbase::vector<int> pre_hashed_key(15);

    int values2[15] = {2, 5, 10, 7, 15, 5, 14, 6, 5, 5, 2, 0, 9, 11, 0};
    for (int i = 0; i < 15; i++) {
        pre_hashed_key(i) = values2[i];
    }

    std::cout << "\nPre-hashed key:" << std::endl;
    for (int i = 0; i < pre_hashed_key.size(); ++i)
        std::cout << pre_hashed_key(i);
    std::cout << "\n";


    // --- Final secret key ---
    libbase::vector<int> final_secret_key(L); // make sure L matches your test

    final_secret_key = pa_system.compute_hashed_key(
        standard_toeplitz_matrix,
        pre_hashed_key,
        L, N, q
    );

    std::cout << "The final secure key = " << final_secret_key << std::endl;
}

BOOST_AUTO_TEST_CASE(testing_pa_standard_toeplitz_without_serialization_using_libbase_gf2)
{
    std::cout << "*** Boost Test 3 ***: Testing Privacy Amplification using Toeplitz Matrix without Serialization directly from object for libbase::gf2" << std::endl;
   // Construct the object directly
    libcomm::pa_standard_toeplitz<libbase::gf2> pa_system;
    const int L = 5;
    const int N = 15;
    const int q = 2;

    /*
    Test 3: Check that I get the same hashed key for the same inputted key

    Results from python script using L = 5, N = 15 and q = 2, gf2 type

    Key to be hashed = [0 1 1 1 0 0 0 0 0 0 0 1 1 1 0]

    (print from toeplitz_hashing.py) generated starting vector = [0 1 0 0 0 1 0 0 0 1 0 0 0 0 1 0 1 1 1]
    (print from base.py) Generated Standard Toeplitz Matrix =
    [[0 1 0 0 0 1 0 0 0 0 1 0 1 1 1]
    [0 0 1 0 0 0 1 0 0 0 0 1 0 1 1]
    [0 0 0 1 0 0 0 1 0 0 0 0 1 0 1]
    [1 0 0 0 1 0 0 0 1 0 0 0 0 1 0]
    [0 1 0 0 0 1 0 0 0 1 0 0 0 0 1]]
    Secure key length: 5
    Generated final secret (hased) key: [1 1 0 1 1]

    Result: Hashed keys matched.

    */
    pa_system.init(L, N, q);

    randgen r;
    r.seed(2602);

    pa_system.seedfrom(r);

    int starting_vector_len = pa_system.generate_starting_vector_length();
    std::cout << "\n Privacy Amplification System Description = " << pa_system.description() << std::endl;
    std::cout << "\nStarting Vector Length = " << starting_vector_len << std::endl;

    libbase::vector<libbase::gf2> starting_vector;
    starting_vector.init(starting_vector_len); // Initialise starting vector with its length.

    // --- Starting vector ---
    // Length = 19
    starting_vector(0)  = 0;
    starting_vector(1)  = 1;
    starting_vector(2)  = 0;
    starting_vector(3)  = 0;
    starting_vector(4)  = 0;
    starting_vector(5)  = 1;
    starting_vector(6)  = 0;
    starting_vector(7)  = 0;
    starting_vector(8)  = 0;
    starting_vector(9)  = 1;
    starting_vector(10) = 0;
    starting_vector(11) = 0;
    starting_vector(12) = 0;
    starting_vector(13) = 0;
    starting_vector(14) = 1;
    starting_vector(15) = 0;
    starting_vector(16) = 1;
    starting_vector(17) = 1;
    starting_vector(18) = 1;

    std::cout << "\nStarting Vector:" << std::endl;
    print_vec_bin(starting_vector, print_gf2_bit);

    libbase::matrix<libbase::gf2> standard_toeplitz_matrix =
        pa_system.generate_toeplitz_matrix(starting_vector);

    std::cout << "\nGenerated Standard Toeplitz Matrix:\n";
    print_mat_bin(standard_toeplitz_matrix, L, N, print_gf2_bit);

    // --- Pre-hashed key ---
    // Length = 15
    libbase::vector<libbase::gf2> pre_hashed_key(15);
    pre_hashed_key(0)  = 0;
    pre_hashed_key(1)  = 1;
    pre_hashed_key(2)  = 1;
    pre_hashed_key(3)  = 1;
    pre_hashed_key(4)  = 0;
    pre_hashed_key(5)  = 0;
    pre_hashed_key(6)  = 0;
    pre_hashed_key(7)  = 0;
    pre_hashed_key(8)  = 0;
    pre_hashed_key(9)  = 0;
    pre_hashed_key(10) = 0;
    pre_hashed_key(11) = 1;
    pre_hashed_key(12) = 1;
    pre_hashed_key(13) = 1;
    pre_hashed_key(14) = 0;

    std::cout << "\nPre-hashed key:" << std::endl;
    print_vec_bin(pre_hashed_key, print_gf2_bit);

    // --- Final secret key ---
    libbase::vector<libbase::gf2> final_secret_key(L); // make sure L matches your test

    final_secret_key = pa_system.compute_hashed_key(
        standard_toeplitz_matrix,
        pre_hashed_key,
        L, N, q
    );

    std::cout << "The final secure key = " << std::endl;
    print_vec_bin(final_secret_key, print_gf2_bit);
}


BOOST_AUTO_TEST_CASE(testing_pa_standard_toeplitz_without_serialization_using_libbase_gf16)
{
    std::cout << "*** Boost Test 4 ***: Testing Privacy Amplification using Toeplitz Matrix without Serialization directly from object for libbase::gf16" << std::endl;
    // Construct the object directly
    libcomm::pa_standard_toeplitz<libbase::gf16> pa_system;
    const int L = 5;
    const int N = 15;
    const int q = 16;

    pa_system.init(L, N, q);

    randgen r;
    r.seed(2602);

    pa_system.seedfrom(r);

    int starting_vector_len = pa_system.generate_starting_vector_length();
    std::cout << "\n Privacy Amplification System Description = " << pa_system.description() << std::endl;
    std::cout << "\nStarting Vector Length = " << starting_vector_len << std::endl;

    /*
    Test 5 : Check that I get the same Toeplitz matrix and Hashed key for libbase::gf16

    L = 5, N = 15, libbase::gf16 type

    Key to be hashed = [ 2  5 10  7 15  5 14  6  5  5  2  0  9 11  0]
    (print from toeplitz_hashing.py) generated starting vector = [ 6  3 12 14 10  7 12  4  6  9  2  6 10 10  7  4  3  7  7]
    (print from base.py) Generated Standard Toeplitz Matrix = [[10  7 12  4  6  9  2  6 10 10  7  4  3  7  7]
    [14 10  7 12  4  6  9  2  6 10 10  7  4  3  7]
    [12 14 10  7 12  4  6  9  2  6 10 10  7  4  3]
    [ 3 12 14 10  7 12  4  6  9  2  6 10 10  7  4]
    [ 6  3 12 14 10  7 12  4  6  9  2  6 10 10  7]]
    Secure key length: 5
    Generated final secret (hased) key: [12  5 12 15  5]

    Results: Worked for libbasee::gf16.
    */

    // Test 4

    // --- Starting vector ---
    libbase::vector<libbase::gf16> starting_vector;
    starting_vector.init(L+N-1);

    int values[19] = {6, 3, 12, 14, 10, 7, 12, 4, 6, 9, 2, 6, 10, 10, 7, 4, 3, 7, 7};

    for (int i = 0; i < 19; i++) {
        starting_vector(i) = values[i];
    }

    std::cout << "\nStarting Vector:" << std::endl;
    print_vec_bin(starting_vector, print_gf16_bin4, /*with_spaces=*/true);

    libbase::matrix<libbase::gf16> standard_toeplitz_matrix =
        pa_system.generate_toeplitz_matrix(starting_vector);

    std::cout << "\nGenerated Standard Toeplitz Matrix:\n";
            // << standard_toeplitz_matrix << std::endl;
    print_mat_bin(standard_toeplitz_matrix, L, N, print_gf16_bin4);

    // --- Pre-hashed key ---
    // Length = 15
    libbase::vector<libbase::gf16> pre_hashed_key(15);

    int values2[15] = {2, 5, 10, 7, 15, 5, 14, 6, 5, 5, 2, 0, 9, 11, 0};
    for (int i = 0; i < 15; i++) {
        pre_hashed_key(i) = values2[i];
    }

    std::cout << "\nPre-hashed key:" << std::endl;
    print_vec_bin(pre_hashed_key, print_gf16_bin4, /*with_spaces=*/true);

    // --- Final secret key ---
    libbase::vector<libbase::gf16> final_secret_key(L); // make sure L matches your test

    final_secret_key = pa_system.compute_hashed_key(
        standard_toeplitz_matrix,
        pre_hashed_key,
        L, N, q
    );

    std::cout << "The final secure key = \n"; //<< final_secret_key << std::endl;
    print_vec_bin(final_secret_key, print_gf16_bin4, /*with_spaces=*/true);
}



BOOST_AUTO_TEST_CASE(testing_pa_standard_toeplitz_with_serialization)
{
    std::cout << "*** Boost Test 5 ***: Testing Privacy Amplification using Toeplitz Matrix with Serialization" << std::endl;

    libcomm::pa_standard_toeplitz<bool> pa_system;

    std::stringstream ss;
    ss << "# Length of final secret hashed key L\n"
    << "5\n"
    << "# Length of pre-hashed key\n"
    << "15\n"
    << "# Alphabet Symbol Size\n"
    << "2\n";

    pa_system.serialize(ss);

    randgen r;
    r.seed(2602);

    pa_system.seedfrom(r);

    std::cout << "\n Privacy Amplification System Description = " << pa_system.description() << std::endl;

    int starting_vector_len = pa_system.generate_starting_vector_length();

    // PA System Parameters
    int N = pa_system.get_N();
    int L = pa_system.get_L();
    int q = pa_system.get_alphabet_size();

    std::cout << "Length of starting vector = " << starting_vector_len << std::endl;

    std::cout << "Length L of the PA system: " << L << std::endl;
    std::cout << "Length N of the PA system: " << N << std::endl;

    std::cout << "Alphabet size of the PA system: " << q << std::endl;

    libbase::vector<bool> starting_vector = pa_system.generate_starting_vector(starting_vector_len, pa_system.get_alphabet_size());

    std::cout << "\nStarting Vector from python:" << std::endl;
    for (int i =0;  i<starting_vector_len; ++i)
    {
        std::cout << starting_vector(i);
    }
    std::cout <<"\n";

    libbase::matrix<bool> standard_toeplitz_matrix = pa_system.generate_toeplitz_matrix(starting_vector);

    std::cout << "\n Generated Standard Toeplitz Matrix:" << standard_toeplitz_matrix  << std::endl;

    libbase::vector<bool> pre_hashed_key(N); // initialises pre-hashed-key
    pre_hashed_key(0) = 1;
    pre_hashed_key(1) = 0;
    pre_hashed_key(2) = 1;
    pre_hashed_key(3) = 1;
    pre_hashed_key(4) = 1;
    pre_hashed_key(5) = 0;
    pre_hashed_key(6) = 1;
    pre_hashed_key(7) = 0;
    pre_hashed_key(8) = 0;
    pre_hashed_key(9) = 0;
    pre_hashed_key(10) = 1;
    pre_hashed_key(11) = 0;
    pre_hashed_key(12) = 0;
    pre_hashed_key(13) = 0;
    pre_hashed_key(14) = 0;

    std::cout << "\nPre-hashed key:" << std::endl;
    for (int i =0;  i<starting_vector_len; ++i)
    {
        std::cout << starting_vector(i);
    }
    std::cout <<"\n";

    libbase::vector<bool> final_secret_key(pa_system.get_L()); // initialises final secret key

    final_secret_key = pa_system.compute_hashed_key(standard_toeplitz_matrix, pre_hashed_key, L, N, q); // computes the hashed key
    std::cout << "\nThe final secure key = " << final_secret_key << std::endl;

}