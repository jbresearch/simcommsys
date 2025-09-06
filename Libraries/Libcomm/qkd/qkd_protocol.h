/*!
 * \file
 *
 * Copyright (c) 2025 Mark Mizzi
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

#ifndef __qkd_protocol_h
#define __qkd_protocol_h

#include "instrumented.h"
#include "qkd/observable.h"
#include "random.h"
#include "serializer.h"
#include "vector.h"
#include <memory>
#include <vector>
#include <string>

#include <type_traits>

namespace libcomm
{

/*!
 * \brief   Common Base for QKD postprocessing protocol.
 * \author  Mark Mizzi
 */
template <typename T, template <class> class C = libbase::vector>
class qkd_protocol : public instrumented, public libbase::serializable
{
public:
    /*! Get observables used to measure quantum states on Alice's end, e.g. spin
     * in two different bases for E91
     * Integer param determines number of observables returned.
     */

    // Note: Changed only the observables to work with a std::vector rather than a libbase::vector
    virtual std::vector<std::unique_ptr<observable<T>>>
    get_alice_observables(int) = 0;

    /*! Get observables used to measure quantum states on Bob's end, e.g. spin
     * in two different bases for E91
     * Integer param determines number of observables returned.
     */

    virtual std::vector<std::unique_ptr<observable<T>>>
    get_alice_observables(int framesize, const libbase::vector<int>&) {
    return get_alice_observables(framesize);}

    virtual std::vector<std::unique_ptr<observable<T>>>
    get_bob_observables(int) = 0;

    virtual const libbase::vector<int>& get_decision_vector() const {
        static libbase::vector<int> empty;
        if (empty.size() == 0) empty.init(0); // libbase::vector needs explicit init
        return empty;
    }

    // Getter to be used to return number of samples for parameter estimation.
    virtual int get_N_PE() = 0;
    virtual int get_N_0() = 0; // to change these. need to check if there is an alternative to = 0? not to force all derived classes?
    virtual double get_v_el() = 0;
    virtual double get_det_eff() = 0;

    // Split fn to be used for parameter estimation and post-processing.
    virtual std::tuple<libbase::vector<double>,
    libbase::vector<double>,
    libbase::vector<double>,
    libbase::vector<double>> split(libbase::vector<double>& measurements_alice, libbase::vector<double>& measurements_bob,
    int N_PE) = 0;

    // Parameter Estimation using Optical Fiber
    virtual std::tuple<double, double, double> parameter_estimation_optical_fiber(libbase::vector<double>& X_PE, libbase::vector<double>& Y_PE, int N_0, double v_el, double detector_efficiency) = 0;

    double calculate_shannon_capacity_awgn(double snr_linear)
    {  // bits/use to be used to compute Beta for MDR.
        return 0.5 * std::log2(1.0 + snr_linear);
    }

    // Channel channel capacity of Quantum Channel.
    virtual double calculate_mutual_information(double chi_total_hat, double VA) = 0;

    virtual double calculate_holevo_bound(double V, double T_hat, double Epsilon_hat, double X_total_hat) = 0;

    // Parameters used to calculate the length of the final secret key after privacy amplification.
    // Note: For the BB84 I would have to also add N_leaked later on.
    virtual void set_parameters_secret_key_length(double beta, double I_AB, double chi_BE, int n_samples) = 0;

    // Equations related to length of final secret key.
    virtual const int calculate_finite_size_effects_secret_key_length() = 0;

    virtual void set_length_secret_key(int l_secret_key) = 0;

    // Helper functions related to codec.
    // virtual std::string codec_description() const = 0;

    // Helper functions related to codec.
    virtual void set_codec(std::shared_ptr<codec<libbase::vector>> CDC) = 0;

    virtual std::shared_ptr<codec<libbase::vector>> get_codec() const = 0;

    virtual  int get_codec_input_bits_k() const = 0;

    virtual C<bool> postprocess(libbase::vector<T>&& alice_measurements,
                                libbase::vector<T>&& bob_measurements) = 0;

    virtual void seedfrom(libbase::random& r) = 0;

    virtual std::string description() const = 0;

    virtual ~qkd_protocol() {}

    DECLARE_BASE_SERIALIZER(qkd_protocol)
};

} // end namespace libcomm

#endif // __qkd_protocol_h