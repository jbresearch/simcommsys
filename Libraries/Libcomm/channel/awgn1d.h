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
 * Expects Eb/N0 in dB via set_parameter(snr_db).
 */
class awgn1d : public channel<double>
{
    // ---- Channel parameters ----
    double snr_db{0.0}; // 10*log10(Eb/N0)
    double Eb{1.0};     // energy per bit
    double No{0.0};     // half the noise energy per symbol
    double sigma{0.0};  // std dev of Gaussian noise: sqrt(Eb*No)

    void compute_noise()
    {
        // No = (N0/2) = 0.5 * 10^(-snr_db/10)
        No = 0.5 * std::pow(10.0, -snr_db / 10.0);
        compute_parameters(Eb, No);
    }

protected:
    void compute_parameters(const double Eb_in, const double No_in);

    double corrupt(const double& s) override;

    double pdf(const double& tx, const double& rx) const override;

public:
    awgn1d() { set_parameter(0.0); } // default 0 dB

    // ---- Parameter API required by mono_parametric ----
    void set_parameter(const double snr_db_in) override
    {
        snr_db = snr_db_in;
        compute_noise();
    }

    double get_parameter() const override { return snr_db; }

    // Convenience (same shape as channel<sigspace>)
    void set_eb(const double Eb_in)
    {
        Eb = Eb_in;
        compute_noise();
    }

    void set_no(const double No_in)
    {
        snr_db = -10.0 * std::log10(2 * No_in);
        compute_noise();
    }

    double get_eb() const { return Eb; }

    double get_no() const { return No; }

    // ---- Description ----
    std::string description() const override;

    // ---- Serialization Support ----
    DECLARE_SERIALIZER(awgn1d)
};

} // namespace libcomm

#endif
