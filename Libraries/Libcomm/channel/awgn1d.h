/*!
 * \file
 *
 * Copyright (c) 2025 Aaron Abela.
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

#ifndef __awgn1d_h
#define __awgn1d_h

#include "channel.h"
#include "config.h"
#include "itfunc.h"
#include "serializer.h"
#include <cmath>
#include <string>

namespace libcomm
{

/*!
 * \brief Additive White Gaussian Noise channel (scalar, 1D) double variant.
 * \author Aaron Abela
 *
 * Expects noise variance via set_parameter(VN).
 */
class awgn1d : public channel<double>
{
    // ---- Channel parameters ----
    double sigma{0.0}; // std dev of Gaussian noise

protected:
    double corrupt(const double& s) override { return s + this->r.gval(sigma); }

    double pdf(const double& tx, const double& rx) const override
    {
        // libbase::gauss expects normalized arg
        return libbase::gauss((rx - tx) / sigma);
    }

public:
    // ---- Parameter API required by mono_parametric ----
    void set_parameter(const double VN) override { sigma = std::sqrt(VN); }

    double get_parameter() const override { return sigma * sigma; }

    // ---- Description ----
    std::string description() const override { return "AWGN channel (scalar)"; }

    // ---- Serialization Support ----
    DECLARE_SERIALIZER(awgn1d)
};

} // namespace libcomm

#endif
