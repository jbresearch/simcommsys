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
    double noise; // additive noise
    double alpha; // fading coefficient

public:
    momentum_observable() : noise(0.0), alpha(0.0) {}
    explicit momentum_observable(double noise_val)
        : noise(noise_val), alpha(0.0)
    {
    }

    void set_noise(double noise_val) { noise = noise_val; }

    double get_noise() const { return noise; }

    void set_alpha(double alpha_val) { alpha = alpha_val; }

    double get_alpha() const { return alpha; }

    double measure(gaussian_state& state) const override
    {
        const double X = state.get_p();
        return alpha * X + noise;
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
