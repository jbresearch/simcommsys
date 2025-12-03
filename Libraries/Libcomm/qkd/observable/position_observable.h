/*!
 * \brief Position Observable
 * \author Aaron Abela
 *
 * Implements a position observable that measures the q-quadrature of a Gaussian
 * quantum state with added noise to simulate realistic measurement.
 */

#ifndef POSITION_OBSERVABLE_H
#define POSITION_OBSERVABLE_H

#include "qkd/observable.h"
#include "qkd/quantum_channel.h"
#include "qkd/quantum_state.h"
#include <cmath>

namespace libcomm
{

// Forward declarations for unused types (optional if not measured)
class qubit;
class entangled_qubit_pair;
class epr_beam;

class position_observable : public observable<double>
{
private:
    double noise;
    double alpha; // fading coefficient 

public:
    position_observable() : noise(0.0), alpha(0.0) {}
    explicit position_observable(double noise_val)
        : noise(noise_val), alpha(0.0)
    {
    }

    /* The noise and alpha are all parameters coming from the gaussian quantum channel. 
    The noise is a CLI parameter of the gaussian quantum channel whereas alpha is a 
    serialized parameter of the gaussian quantum channel.
    */
   
    void set_noise(double noise_val) { noise = noise_val; }

    double get_noise() const { return noise; }

    void set_alpha(double alpha_val)
    {
        alpha = alpha_val;
    }

    double get_alpha() const { return alpha; }

    // Method that does the measurement on a gaussian coherent state
    double measure(gaussian_state& state) const override
    {
        // return std::sqrt(transmittance*detector_eff)*(state.get_q() + noise);
        // // For the case 1: S_B\ =\ \sqrt\etaT\left(S_A\ \ +\ \ S_N\right):

        const double X = state.get_q();
        const double g = alpha;
        const double gX = g * X;
        const double n = noise; // Noise is being set in the transmit method of
                                // the gaussian quantum channel.
        const double result = gX + n;

        // Debug prints
        // std::cout << "\n**** Breakdown of values for PE step from
        // position_observable.h: ****\n"; std::cout << "X = " << X << "\n";
        // std::cout << "sqrt(eta*T) * X = " << gX << "\n";
        // std::cout << "noise = " << n << "\n";
        // std::cout << "result = " << result << "\n";

        return result;

        // return (std::sqrt(transmittance*detector_eff)*(X)) + noise; // For
        // case 2: S_B\ =\ \sqrt\etaT\left(S_A\ \right)\ +\ S_N:

        // return state.get_q() + noise; // For case 3: S_B\ =\ S_A+\ S_N
        // return (transmittance*detector_eff)*(state.get_q() + noise); // case
        // 7
    }

    // To double check with johann whether quantum_channel should be a const or
    // not
    void transmit(quantum_channel& c) override
    {
        c.transmit(*this); // Double dispatch: calls
                           // quantum_channel::transmit(position_observable&)
    }
    // void transmit(const quantum_channel& c) { return c.transmit(*this); }

    double measure(qubit&) const override
    {
        failwith("position_observable does not support qubit measurement.");
        return 0;
    }

    double measure(entangled_qubit_pair&, int) const override
    {
        failwith("position_observable does not support entangled qubit "
                 "measurement.");
        return 0;
    }

    double measure(epr_beam&, int) const override
    {
        failwith("position_observable does not support EPR beam measurement.");
        return 0;
    }
};

} // namespace libcomm

#endif // POSITION_OBSERVABLE_H
