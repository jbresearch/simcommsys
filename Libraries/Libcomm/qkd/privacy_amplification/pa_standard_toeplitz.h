
/*!
 * Documentation of Code:
 *
 * This code is based on the Python implementation of the toeplitz_standard.py
 * implemented by Dr. Ing. Chris Galea found in the following repository which
 * uses Scipy:
 * https://dsrg-ict.research.um.edu.mt/qkd/ldpc-codes/-/blob/modularise_hungary_code/Other/Botond/Code/privacy_amplification/toeplitz_standard.py?ref_type=heads
 *
 * The Toeplitz matrix is built from a single starting vector.
 *
 * sv is the starting_vector which will be used to build the toeplitz matrix.
 *
 * L is the length of the final hashed key outputted after privacy
 * amplification.
 *
 * N is the length of the inputted key that will be hashed. In the case of MDR
 * the input vector is Bob's s vector of size k.
 *
 */

#ifndef __pa_standard_toeplitz_h
#define __pa_standard_toeplitz_h

#include "assertalways.h"
#include "qkd/privacy_amplification.h"
#include "toeplitz_standard.h"
#include <string>

namespace libcomm
{

template <class T>
class pa_standard_toeplitz : public privacy_amplification_base<T>
{
private:
    int L;             // Final length of secret key
    int N;             // Length of pre-hashed key.
    int alphabet_size; // e.g. 2 for binary arithmetic.

public:
    void init(int L_, int N_, int q_)
    {
        L = L_;
        N = N_;
        alphabet_size = q_;
    }

    void seedfrom(libbase::random& r) override { this->rng.seed(r.ival()); }

    int generate_starting_vector_length() override
    {
        assertalways(L > 0 && N > 0);
        return L + N - 1; // Length of starting vector.
    }

    libbase::matrix<T>
    generate_toeplitz_matrix(const libbase::vector<T>& starting_vector) override
    {
        assertalways(L > 0 && N > 0);
        assertalways(starting_vector.size() == L + N - 1);

        libbase::matrix<T> standard_Toeplitz =
            libbase::toeplitz_standard::build<T>(starting_vector, L, N);

        return standard_Toeplitz;
    }

    // Description function
    std::string description() const override
    {
        std::ostringstream sout;
        sout << "Standard Toeplitz Matrix (L=" << L << ", N=" << N
             << ", q=" << alphabet_size << ")";
        return sout.str();
    }

    // getters to get the sizes of L, N and alphabet_size
    int get_L() { return L; }
    int get_N() { return N; }
    int get_alphabet_size() { return alphabet_size; }

    // setters to set the sizes of L, N and alphabet_size
    void set_L(int l) { this->L = l; }
    void set_N(int n) { this->N = n; }
    void set_alphabet_size(int q) { this->alphabet_size = q; }

    DECLARE_SERIALIZER(pa_standard_toeplitz<T>)
};

} // namespace libcomm

#endif // __pa_standard_toeplitz_h