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

#ifndef __config_h
#define __config_h

/*!
 * \file
 * \brief   Main Configuration.
 * \author  Johann Briffa
 */

// system include files - all architectures

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <stdint.h>

// module include files

#include "assertalways.h"

// *** Within library namespace ***

namespace libbase
{

// Debugging tools

extern std::ostream trace;

// Constants

extern const double PI;

// Interactive keyboard handling
int keypressed(void);
int readkey(void);

// Interrupt-signal handling function
bool interrupted(void);

// System error message reporting
std::string getlasterror();

// Functions to skip over whitespace and comments
std::istream& eatwhite(std::istream& is);
std::istream& eatcomments(std::istream& is);

// Exception class for stream load errors
class load_error : public std::runtime_error
{
public:
    explicit load_error(const std::string& what_arg)
        : std::runtime_error(what_arg)
    {
    }
};

// Stream data loading verification functions
void check_failedload(std::istream& is);
void check_incompleteload(std::istream& is);
std::istream& verify(std::istream& is);
std::istream& verifycomplete(std::istream& is);

// Check for alignment
inline bool
isaligned(const void* buf, int bytes)
{
    return ((long)buf & (bytes - 1)) == 0;
}

} // namespace libbase

// *** Within standard library namespace ***

namespace std
{

//! Operator to concatenate STL vectors
template <class T>
void
operator+=(std::vector<T>& a, const std::vector<T>& b)
{
    a.insert(a.end(), b.begin(), b.end());
}

/*! \brief Serialize STL vectors
 *
 * \note This was needed for use of \c multitoken args in boost \c
 * program_options , but it may be useful elsewhere
 */
template <class T>
ostream&
operator<<(ostream& os, const std::vector<T>& xs)
{
    os << xs.size();
    if (xs.size() > 0) {
        os << std::endl;
        for (auto it = xs.begin(); it != --xs.end(); ++it) {
            os << *it << '\t';
        }
        os << *--xs.end() << std::endl;
    }
    return os << std::flush;
}

/*! \brief De-serialize STL vectors
 *
 * \note This was needed for use of \c multitoken args in boost \c
 * program_options , but it may be useful elsewhere
 */
template <class T>
istream&
operator>>(istream& is, std::vector<T>& xs)
{
    xs.clear();
    size_t len;
    is >> libbase::eatcomments >> len >> libbase::verify;
    for (size_t i = 0; i < len; i++) {
        T x;
        is >> libbase::eatcomments >> x >> libbase::verify;
        // make sure we move for T that are expensive to copy.
        xs.push_back(std::move(x));
    }
    return is;
}

} // namespace std

#endif
