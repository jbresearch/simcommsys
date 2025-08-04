/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
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

#include "bitfield.h"
#include "cputimer.h"
#include "gf.h"
#include "matrix.h"
#include <boost/preprocessor/seq/for_each.hpp>
#include <cstdint>
#include <iostream>

using libbase::bitfield;
using libbase::cputimer;
using libbase::gf;
using libbase::matrix;

using std::cerr;
using std::cout;
using std::dec;
using std::hex;

//! \brief Basic impl. of + to ensure correctness of more sophisticated impl.
template <int m, int poly>
uint32_t
gf_test_add(uint32_t x, uint32_t y)
{
    return x ^ y;
}

//! \brief Basic impl. of * to ensure correctness of more sophisticated impl.
template <int m, int poly>
uint32_t
gf_test_mul(uint32_t x, uint32_t y)
{
    // Initialize result
    uint32_t res = 0;
    // Loop over all bits in multiplicand
    for (int i = 0; i < m && y != 0; i++) {
        // If the corresponding bit in the multiplicand is set,
        // add (XOR) the shifted multiplier
        if (y & 1) {
            res ^= x;
        }

        // Shift the multiplicand
        y >>= 1;
        // Shift the multiplier, subtracting the polynomial on overflow
        x <<= 1;

        if (x & (1 << m)) {
            x ^= poly;
        }
    }

    return res;
}

//! \brief Basic impl. of inverse() to ensure correctness of more sophisticated
//! impl.
template <int m, int poly>
uint32_t
gf_test_inverse(uint32_t x)
{
    using libbase::gf;

    gf<m, poly> result(1);
    for (int i = 1; i < gf<m, poly>::elements(); i++) {
        if (result * gf<m, poly>(x) == gf<m, poly>(1)) {
            break;
        }
        result *= gf<m, poly>(2);
    }
    return uint32_t(result);
}

/*!
 * \brief Exponential table entries for base {03}
 * cf. Gladman, "A Specification for Rijndael, the AES Algorithm", 2003, p.5
 */
const int aestable[] = {
    0x01, 0x03, 0x05, 0x0f, 0x11, 0x33, 0x55, 0xff, 0x1a, 0x2e, 0x72, 0x96,
    0xa1, 0xf8, 0x13, 0x35, 0x5f, 0xe1, 0x38, 0x48, 0xd8, 0x73, 0x95, 0xa4,
    0xf7, 0x02, 0x06, 0x0a, 0x1e, 0x22, 0x66, 0xaa, 0xe5, 0x34, 0x5c, 0xe4,
    0x37, 0x59, 0xeb, 0x26, 0x6a, 0xbe, 0xd9, 0x70, 0x90, 0xab, 0xe6, 0x31,
    0x53, 0xf5, 0x04, 0x0c, 0x14, 0x3c, 0x44, 0xcc, 0x4f, 0xd1, 0x68, 0xb8,
    0xd3, 0x6e, 0xb2, 0xcd, 0x4c, 0xd4, 0x67, 0xa9, 0xe0, 0x3b, 0x4d, 0xd7,
    0x62, 0xa6, 0xf1, 0x08, 0x18, 0x28, 0x78, 0x88, 0x83, 0x9e, 0xb9, 0xd0,
    0x6b, 0xbd, 0xdc, 0x7f, 0x81, 0x98, 0xb3, 0xce, 0x49, 0xdb, 0x76, 0x9a,
    0xb5, 0xc4, 0x57, 0xf9, 0x10, 0x30, 0x50, 0xf0, 0x0b, 0x1d, 0x27, 0x69,
    0xbb, 0xd6, 0x61, 0xa3, 0xfe, 0x19, 0x2b, 0x7d, 0x87, 0x92, 0xad, 0xec,
    0x2f, 0x71, 0x93, 0xae, 0xe9, 0x20, 0x60, 0xa0, 0xfb, 0x16, 0x3a, 0x4e,
    0xd2, 0x6d, 0xb7, 0xc2, 0x5d, 0xe7, 0x32, 0x56, 0xfa, 0x15, 0x3f, 0x41,
    0xc3, 0x5e, 0xe2, 0x3d, 0x47, 0xc9, 0x40, 0xc0, 0x5b, 0xed, 0x2c, 0x74,
    0x9c, 0xbf, 0xda, 0x75, 0x9f, 0xba, 0xd5, 0x64, 0xac, 0xef, 0x2a, 0x7e,
    0x82, 0x9d, 0xbc, 0xdf, 0x7a, 0x8e, 0x89, 0x80, 0x9b, 0xb6, 0xc1, 0x58,
    0xe8, 0x23, 0x65, 0xaf, 0xea, 0x25, 0x6f, 0xb1, 0xc8, 0x43, 0xc5, 0x54,
    0xfc, 0x1f, 0x21, 0x63, 0xa5, 0xf4, 0x07, 0x09, 0x1b, 0x2d, 0x77, 0x99,
    0xb0, 0xcb, 0x46, 0xca, 0x45, 0xcf, 0x4a, 0xde, 0x79, 0x8b, 0x86, 0x91,
    0xa8, 0xe3, 0x3e, 0x42, 0xc6, 0x51, 0xf3, 0x0e, 0x12, 0x36, 0x5a, 0xee,
    0x29, 0x7b, 0x8d, 0x8c, 0x8f, 0x8a, 0x85, 0x94, 0xa7, 0xf2, 0x0d, 0x17,
    0x39, 0x4b, 0xdd, 0x7c, 0x84, 0x97, 0xa2, 0xfd, 0x1c, 0x24, 0x6c, 0xb4,
    0xc7, 0x52, 0xf6, 0x01};

template <typename GF_q>
void
TestField()
{
    constexpr int m = GF_q::dimension();
    constexpr int poly = GF_q::polynomial();

    // we can't use templated functions in assert() as it complains
    auto gf_test_add_ = [](auto x, auto y) {
        return gf_test_add<m, poly>(x, y);
    };
    auto gf_test_mul_ = [](auto x, auto y) {
        return gf_test_mul<m, poly>(x, y);
    };
    auto gf_test_inv_ = [](auto x) { return gf_test_inverse<m, poly>(x); };

    // Test addition and mul. against basic impl.
    std::cout << "Testing addition for " << GF_q(0).description() << std::endl;
    for (int x = 0; x < GF_q::elements(); x++) {
        for (int y = 0; y < GF_q::elements(); y++) {
            assertalways(gf_test_add_(x, y) == uint32_t(GF_q(x) + GF_q(y)));
        }
    }

    std::cout << "Testing multiplication for " << GF_q(0).description()
              << std::endl;
    for (int x = 0; x < GF_q::elements(); x++) {
        for (int y = 0; y < GF_q::elements(); y++) {
            assertalways(gf_test_mul_(x, y) == uint32_t(GF_q(x) * GF_q(y)));
        }
    }

    std::cout << "Testing inverse for " << GF_q(0).description() << std::endl;
    for (int x = 0; x < GF_q::elements(); x++) {
        assertalways(gf_test_inv_(x) == uint32_t(GF_q(x).inverse()));
    }
}

void
TestRijndaelField()
{
    // Create a value in the Rijndael field GF(2^8): m(x) = 1 { 0001 1011 }
    gf<8, 0x11B> E = 1;
    // Compute and display exponential table using {03} as a multiplier
    // using the tabular format in Gladman.
    cout << std::endl << "Rijndael GF(2^8) exponentiation table:" << std::endl;
    cout << hex;
    for (int x = 0; x < 16; x++) {
        for (int y = 0; y < 16; y++) {
            assert(E == aestable[(x << 4) + y]);
            cout << int(E) << (y == 15 ? '\n' : '\t');
            E *= 3;
        }
    }
    cout << dec;
}

template <int m, int poly>
void
ListField()
{
    // Compute and display exponential table using {2} as a multiplier
    cout << std::endl
         << "GF(" << m << ",0x" << hex << poly << dec
         << ") table:" << std::endl;
    cout << 0 << '\t' << 0 << '\t' << bitfield(0, m) << std::endl;
    gf<m, poly> E = 1;
    for (int x = 1; x < gf<m, poly>::elements(); x++) {
        cout << x << "\ta" << x - 1 << '\t' << bitfield(E, m) << std::endl;
        E *= 2;
    }
}

template <int m, int poly>
void
TestMulDiv()
{
    // Compute and display exponential table using {2} as a multiplier
    cout << std::endl
         << "GF(" << m << ",0x" << hex << poly << dec
         << ") multiplication/division:" << std::endl;
    cout << "power\tvalue\tinverse\tmul" << std::endl;
    gf<m, poly> E = 1;
    for (int x = 1; x < gf<m, poly>::elements(); x++) {
        cout << "a" << x - 1 << '\t' << bitfield(E, m) << '\t'
             << bitfield(E.inverse(), m) << '\t' << bitfield(E.inverse() * E, m)
             << std::endl;
        E *= 2;
    }
}

void
TestGenPowerGF2()
{
    cout << std::endl << "Binary generator matrix power sequence:" << std::endl;
    // Create values in the Binary field GF(2): m(x) = 1 { 1 }
    typedef gf<1, 0x3> Binary;
    // Create generator matrix for DVB-CRSC code:
    matrix<Binary> G(3, 3);
    G = 0;
    G(0, 0) = 1;
    G(2, 0) = 1;
    G(0, 1) = 1;
    G(1, 2) = 1;
    // Compute and display first 8 powers of G
    for (int i = 0; i < 8; i++) {
        cout << "G^" << i << " = " << std::endl;
        pow(G, i).serialize(cout);
    }
}

void
TestGenPowerGF8()
{
    cout << std::endl << "GF(8) generator matrix power sequence:" << std::endl;
    // Create values in the field GF(8): m(x) = 1 { 011 }
    typedef gf<3, 0xB> GF8;
    // Create generator matrix:
    matrix<GF8> G(2, 2);
    G(0, 0) = 1;
    G(1, 0) = 6;
    G(0, 1) = 1;
    G(1, 1) = 0;
    // Compute and display first 16 powers of G
    for (int i = 0; i < 16; i++) {
        cout << "G^" << i << " = " << std::endl;
        pow(G, i).serialize(cout);
    }
}

/*!
 * \brief Test program for GF class
 * \author  Johann Briffa
 */

int
main(int argc, char* argv[])
{
    // test each of the fields in GF_TYPE_SEQ
#define TESTFIELD(r, x, type) TestField<libbase::type>();
    BOOST_PP_SEQ_FOR_EACH(TESTFIELD, x, GF_TYPE_SEQ)

    TestRijndaelField();
    ListField<2, 0x7>();
    ListField<3, 0xB>();
    ListField<4, 0x13>();
    TestMulDiv<3, 0xB>();
    TestGenPowerGF2();
    TestGenPowerGF8();
    return 0;
}
