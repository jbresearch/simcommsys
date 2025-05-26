/*!
 * \brief Position Observable
 * \author  Aaron Abela
 *
 * Allows creation of a position observable with a specific value q.
 */


#ifndef POSITION_OBSERVABLE_H
#define POSITION_OBSERVABLE_H

namespace libcomm {

class position_observable {
public:
    double value;

    position_observable() : value(0.0) {}
    position_observable(double q) : value(q) {}
};

} // namespace libcomm

#endif