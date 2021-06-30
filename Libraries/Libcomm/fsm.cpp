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

#include "fsm.h"

namespace libcomm
{

const int fsm::tail = -1;

int fsm::convert(const array1i_t& vec, int S)
{
    const int nu = vec.size();
    assert(pow(S, nu) - 1 <= std::numeric_limits<int>::max());
    int val = 0;

    for (int i = nu - 1; i >= 0; i--) {
        val *= S;
        assert(vec(i) >= 0 && vec(i) < S);
        val += vec(i);
    }

    return val;
}

fsm::array1i_t fsm::convert(int val, int nu, int S)
{
    array1i_t vec(nu);
    assert(val >= 0);

    for (int i = 0; i < nu; i++) {
        vec(i) = val % S;
        val /= S;
    }

    assert(val == 0);
    return vec;
}

int fsm::convert_input(const array1i_t& vec) const
{
    assert(vec.size() == num_inputs());
    return convert(vec, num_symbols());
}

fsm::array1i_t fsm::convert_input(int val) const
{
    return convert(val, num_inputs(), num_symbols());
}

int fsm::convert_output(const array1i_t& vec) const
{
    assert(vec.size() == num_outputs());
    return convert(vec, num_symbols());
}

fsm::array1i_t fsm::convert_output(int val) const
{
    return convert(val, num_outputs(), num_symbols());
}

int fsm::convert_state(const array1i_t& vec) const
{
    assert(vec.size() == mem_elements());
    return convert(vec, num_symbols());
}

fsm::array1i_t fsm::convert_state(int val) const
{
    return convert(val, mem_elements(), num_symbols());
}

bool fsm::can_be_cached() const
{
    if (pow(num_symbols(), num_outputs()) - 1 >
        std::numeric_limits<int>::max()) {
        return false;
    }

    if (num_states() > std::numeric_limits<int>::max()) {
        return false;
    }

    if (num_input_combinations() > std::numeric_limits<int>::max()) {
        return false;
    }

    if (num_output_combinations() > std::numeric_limits<int>::max()) {
        return false;
    }

    return true;
}

} // namespace libcomm
