/*!
 * \file pa_standard_toeplitz.cpp
 *
 * Copyright (c) 2025 Aaron Abela
 *
 * This file is part of SimCommSys.
 * Released under the GNU General Public License v3 or later.
 */

#include "pa_standard_toeplitz.h"
#include "serializer.h"
#include <iostream>
#include <sstream>
#include <string>

using libbase::serializer;
namespace libcomm
{
    // left to do are the serialization fns

    template<class T>
    std::ostream& pa_standard_toeplitz<T>::serialize(std::ostream& sout) const
    {
        std::ostringstream sout;
        sout << "Length of final secret hashed key L" << std::endl;
        sout << L << std::endl;
        sout << "Length of pre-hashed key" << std::endl;
        sout << N << std::endl;
        sout << "Alphabet Symbol Size" << std::endl;
        sout << alphabet_size << std::endl;
        return sout;
    }

    template<class T>
    std::istream& pa_standard_toeplitz<T>::serialize(std::istream& sin)
    {
        assertalways(sin.good());
        sin >> libbase::eatcomments >> L >> libbase::verify;
        sin >> libbase::eatcomments >> N >> libbase::verify;
        sin >> libbase::eatcomments >> alphabet_size >> libbase::verify;
        return sin;
    }

    template<class T>
    const serializer pa_standard_toeplitz<T>::shelper("privacy_amplification", "standard_toeplitz", pa_standard_toeplitz<T>::create);


    // Explicit instantiations for used types T:
    template class pa_standard_toeplitz<bool>;
    template class pa_standard_toeplitz<int>;

}
