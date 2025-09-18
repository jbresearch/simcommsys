/*!
 * \brief Momentum Observable
 * \author Aaron Abela
 *
 * Implements a momentum observable that measures the p-quadrature of a Gaussian
 * quantum state with added noise to simulate realistic measurement.
 */

#ifndef MOMENTUM_OBSERVABLE_H
#define MOMENTUM_OBSERVABLE_H

#include "qkd/observable.h"
#include "qkd/quantum_channel.h"
#include "qkd/quantum_state.h"

namespace libcomm
{

class qubit;
class entangled_qubit_pair;
class epr_beam;

class momentum_observable : public observable<double>
{
private:
    double noise;
    double transmittance;
    double detector_eff;

public:
    momentum_observable() : noise(0.0), transmittance(0.0), detector_eff(0.0) {}
    explicit momentum_observable(double noise_val)
        : noise(noise_val), transmittance(0.0), detector_eff(0.0)
    {
    }

    void set_noise(double noise_val) { noise = noise_val; }

    double get_noise() const { return noise; }

    void set_transmittance(double transmittance_val)
    {
        transmittance = transmittance_val;
    }

    double get_transmittance() const { return transmittance; }

    void set_detector_eff(double detector_eff_val)
    {
        detector_eff = detector_eff_val;
    }

    double get_detector_eff() const { return detector_eff; }

    double measure(gaussian_state& state) const override
    {
        // return std::sqrt(transmittance*detector_eff)*(state.get_p() + noise);
        // // For the case 1: S_B\ =\ \sqrt\etaT\left(S_A\ \ +\ \ S_N\right):

        const double X = state.get_p();
        const double g = std::sqrt(transmittance * detector_eff);
        const double gX = g * X;
        const double n = noise;
        const double result = gX + n;

        // // Debug prints
        // std::cout << "\n**** Breakdown of values for PE step from
        // momentum_observable.h: ****\n"; std::cout << "X = " << X << "\n";
        // std::cout << "sqrt(eta*T) * X = " << gX << "\n";
        // std::cout << "noise = " << n << "\n";
        // std::cout << "result = " << result << "\n";

        return result;

        // return (std::sqrt(transmittance*detector_eff)*(X)) + noise; // For
        // case 2: S_B\ =\ \sqrt\etaT\left(S_A\ \right)\ +\ S_N:

        // return state.get_p() + noise; // For case 3: S_B\ =\ S_A+\ S_N
        // return (transmittance*detector_eff)*(state.get_p() + noise); // case
        // 7
    }

    // Same as the position observable to double check with Johann if it should
    // be a const or not.
    void transmit(quantum_channel& c) override
    {
        c.transmit(*this); // Double dispatch: calls
                           // quantum_channel::transmit(momentum_observable&)
    }

    // Helper functions
    double measure(qubit&) const override
    {
        failwith("momentum_observable does not support qubit measurement.");
        return 0;
    }

    double measure(entangled_qubit_pair&, int) const override
    {
        failwith("momentum_observable does not support entangled qubit "
                 "measurement.");
        return 0;
    }

    double measure(epr_beam&, int) const override
    {
        failwith("momentum_observable does not support EPR beam measurement.");
        return 0;
    }
};

} // namespace libcomm

#endif // MOMENTUM_OBSERVABLE_H
