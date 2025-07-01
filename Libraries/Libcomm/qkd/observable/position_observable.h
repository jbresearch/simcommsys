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

namespace libcomm {

// Forward declarations for unused types (optional if not measured)
class qubit;
class entangled_qubit_pair;
class epr_beam;

class position_observable : public observable<double> {
private:
    double noise;

public:
    position_observable() : noise(0.0) {}
    explicit position_observable(double noise_val) : noise(noise_val) {}

    void set_noise(double noise_val) {
        noise = noise_val;
    }

    double get_noise() const {
        return noise;
    }

    double measure(gaussian_state& state) const override {
        return state.get_q() + noise;
    }

    // To double check with johann whether quantum_channel should be a const or not
    void transmit(quantum_channel& c) override {
        c.transmit(*this);  // Double dispatch: calls quantum_channel::transmit(position_observable&)
    }

    // void transmit(const quantum_channel& c) { return c.transmit(*this); }

    // Helper functions
    double measure(qubit&) const override {
        failwith("position_observable does not support qubit measurement.");
        return 0;
    }

    double measure(entangled_qubit_pair&, int) const override {
        failwith("position_observable does not support entangled qubit measurement.");
        return 0;
    }

    double measure(epr_beam&, int) const override {
        failwith("position_observable does not support EPR beam measurement.");
        return 0;
    }
};

} // namespace libcomm

#endif // POSITION_OBSERVABLE_H
