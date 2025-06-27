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

#ifndef __quantum_channel_h
#define __quantum_channel_h

#include "assertalways.h"
#include "parametric.h"
#include "random.h"
#include "serializer.h"
#include <random>
#include <cmath>


namespace libcomm
{

class position_observable;
class momentum_observable;

class spin_computational;
class spin_hadamard;

/*!
 * \brief   Common Base for Quantum channel.
 * \author  Mark Mizzi
 *
 * We use the Heisenberg picture, so that quantum channels act on observables.
 * We use the visitor pattern to model these actions, so the base interface
 * contains methods (called \ref transmit()) for "visiting" each concrete
 * observable type. By default these methods fail, but a concrete quantum
 * channel subclass is meant to override the methods for observables that they
 * support.
 */
class quantum_channel : public parametric, public libbase::serializable
{
public:
    //! \name Visitor interface methods for observables
    virtual void transmit(position_observable&) const
    {
        failwith("Not implemented.");
    }
    virtual void transmit(momentum_observable&) const
    {
        failwith("Not implemented.");
    }
    virtual void transmit(spin_computational&) const
    {
        failwith("Not implemented.");
    }
    virtual void transmit(spin_hadamard&) const
    {
        failwith("Not implemented.");
    }
    //! @}

    //! \brief Description of serializable object
    virtual std::string description() const = 0;

    // Serialization Support
    DECLARE_BASE_SERIALIZER(quantum_channel)

    virtual void seedfrom(libbase::random& r) = 0;
    virtual ~quantum_channel() {}
};

/*!
 * \brief   Gaussian Quantum channel.
 * \author  Aaron Abela
 *
 * Supports every observable.
 * Noise is modelled as a Normal Distribution with Mean zero and Variance V_N:
 *     V_N = N_0 + ηTξ + v_el -> Taken from "Quasi-cyclic multi-edge LDPC codes for long-distance quantum cryptography." by Mario Milicevic et al., 2018, npj Quantum Information"
 * Bob's measured value:
 *     X_B = sqrt(ηT) * (X_A + X_N) -> Taken from eq. (7.33)  -204 -  from the book "Quantum Key Distribution" by Ramona Wolf, Springer
 */
class gaussian_quantum_channel : public quantum_channel
{
private:
    double N_0;              // Vacuum noise
    double det_eff;          // Detector efficiency (η)
    mutable double transmittance;    // Transmittance T
    double distance_km;     // Distance l in km
    double excess_noise;    // Excess noise (ξ)
    double electric_noise;   //  Electronic noise (v_el)
    double alpha;           // Attenuation coefficient (α)
    mutable double V_N; // Total noise variance V_N used in the gaussian quantum channel

    mutable std::mt19937 gen;

    void update_channel_properties() const {
        transmittance = calculate_transmittance();
        V_N = calculate_variance_v_n();
    }

public:
    double calculate_transmittance() const {
        return std::pow(10.0, -alpha * distance_km / 10.0);
    }

    double calculate_variance_v_n() const {
        return N_0 + det_eff * transmittance * excess_noise + electric_noise;
    }

    // position observable - p; STILL TO REDO AND recheck
    void transmit(position_observable& obs) const override {
        // std::normal_distribution<double> dist(0.0, std::sqrt(V_N));
        // double noise = dist(gen) * std::sqrt(transmittance * det_eff);
        // obs.set_noise(noise);  // ✅ the new design uses this method
         failwith("Not implemented.");
    }


    // momentum observable - p; Does not override the base class
    void transmit(momentum_observable& obs) const override {
        // std::normal_distribution<double> dist(0.0, std::sqrt(V_N));
        // obs.value += dist(gen); // Adds X_N
        // obs.value *= std::sqrt(transmittance * det_eff); // Scales with sqrt(ηT)
         failwith("Not implemented.");
    }

    void seedfrom(libbase::random& r) override {
    // Non-deterministic seed using system entropy/entropy based seed.
    gen.seed(std::random_device{}()); // to change this like I did in quantum_gaussian.h
    }

    void set_parameters(const libbase::vector<double>& x) override {
        assertalways(x.size() == 6);  // Ensures all required parameters are passed
        N_0 = x(0);
        alpha = x(1);
        det_eff = x(2);
        distance_km = x(3);
        excess_noise = x(4);
        electric_noise = x(5);
        update_channel_properties();
    }


    libbase::vector<double> get_parameters() const override {
        libbase::vector<double> params;
        params.init(8);
        params(0) = N_0;
        params(1) = alpha;
        params(2) = det_eff;
        params(3) = distance_km;
        params(4) = excess_noise;
        params(5) = electric_noise;
        params(6) = transmittance;
        params(7) = V_N;
        return params;
    }

    int get_num_params() const override { return 6; }

    // Helper function
    void print_parameters(std::ostream& os = std::cout) const {
    os << "Gaussian Quantum Channel Parameters:\n"
       << "  N_0              = " << N_0 << "\n"
       << "  alpha            = " << alpha << "\n"
       << "  detector_eff     = " << det_eff << "\n"
       << "  distance_km      = " << distance_km << "\n"
       << "  excess_noise     = " << excess_noise << "\n"
       << "  electric_noise   = " << electric_noise << "\n"
       << "  transmittance    = " << transmittance << "\n"
       << "  V_N              = " << V_N << "\n";
    }

    // Required for TestGaussianCVQKDsource with Boost usage
    static std::unique_ptr<libbase::serializable> create(std::istream& sin) {
        auto obj = std::make_unique<gaussian_quantum_channel>();
        obj->serialize(sin);
        return obj;
    }

    std::string description() const override;

    DECLARE_SERIALIZER(gaussian_quantum_channel)
};


} // end namespace libcomm

#endif // __quantum_channel_h