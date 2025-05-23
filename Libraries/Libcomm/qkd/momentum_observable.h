#ifndef MOMENTUM_OBSERVABLE_H
#define MOMENTUM_OBSERVABLE_H

namespace libcomm {

class momentum_observable {
public:
    double value;

    momentum_observable() : value(0.0) {}
    momentum_observable(double p) : value(p) {}
};

} // namespace libcomm

#endif
