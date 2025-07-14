/*!
 * \brief Fake Momentum Observable (Noiseless)
 * \author Aaron Abela
 *
 * A simplified observable that measures the p-quadrature of a Gaussian
 * quantum state with perfect transmittance and no noise or inefficiency which will only be utilised for Alice.
 */

#ifndef FAKE_MOMENTUM_OBSERVABLE_H
#define FAKE_MOMENTUM_OBSERVABLE_H

#include "qkd/observable.h"
#include "qkd/quantum_channel.h"
#include "qkd/quantum_state.h"
#include <cmath>

namespace libcomm {

class qubit;
class entangled_qubit_pair;
class epr_beam;

class fake_momentum_observable : public observable<double> {
public:
    fake_momentum_observable() {}

    // Perfect measurement with no noise or loss
    double measure(gaussian_state& state) const override {
        return state.get_p_mean();
        return 0;
    }

    void transmit(quantum_channel& c) override {
        c.transmit(*this);
    }

    double measure(qubit&) const override {
        failwith("fake_momentum_observable does not support qubit measurement.");
        return 0;
    }

    double measure(entangled_qubit_pair&, int) const override {
        failwith("fake_momentum_observable does not support entangled qubit pair measurement.");
        return 0;
    }

    double measure(epr_beam&, int) const override {
        failwith("fake_momentum_observable does not support EPR beam measurement.");
        return 0;
    }
};

} // namespace libcomm

#endif // FAKE_MOMENTUM_OBSERVABLE_H
