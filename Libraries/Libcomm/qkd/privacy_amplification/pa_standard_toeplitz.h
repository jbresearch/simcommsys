#ifndef __pa_standard_toeplitz_h
#define __pa_standard_toeplitz_h

#include "qkd/privacy_amplification.h"
#include "toeplitz_standard.h"
#include "assertalways.h"
#include <string>

namespace libcomm {

template<class T>
class pa_standard_toeplitz : public privacy_amplification<T>
{
    private:
        int L; // Final length of secret key
        int N; // Length of pre-hashed key.
        libbase::randgen rng;
        int alphabet_size;    // e.g. 2 for binary arithmetic.

    public:

        void seedfrom(libbase::random& r) override { this->rng.seed(r.ival()); }

        int generate_starting_vector_length() override
        {
            assertalways(L > 0 && N > 0);
            return L + N - 1; // Length of starting vector.
        }

        libbase::matrix<T> generate_toeplitz_matrix(const libbase::vector<T>& starting_vector) override
        {
            assertalways(L > 0 && N > 0);
            assertalways(starting_vector.size() == L + N - 1);

            libbase::matrix<T> standard_Toeplitz = libbase::toeplitz_standard::build<T>(starting_vector, L, N);

            return standard_Toeplitz;
        }

        // Description function
        std::string description() const override
        {
            return "Privacy Amplification using the standard Toeplitz matrix";
        }

        DECLARE_SERIALIZER(pa_standard_toeplitz<T>)
};

}

#endif // __pa_standard_toeplitz_h