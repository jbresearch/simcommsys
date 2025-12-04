#ifndef __depolarizing_quantum_channel_h
#define __depolarizing_quantum_channel_h

#include "qkd/observable/hadamard_observable.h"
#include "qkd/observable/computational_observable.h"
#include "qkd/quantum_channel.h"
#include <cmath>
#include <random>

namespace libcomm
{

/*!
 * \brief   A simple depolarizing channel for BB84.
 * \author  Aaron Abela
 *
 * This channel models a simple, basis-independent Quantum Bit Error Rate (QBER) for the BB84 protocol.
 * It passes this QBER to the observables, which then apply the noise during measurement.
 *
 * Reference: Thakur, V.S., Kumar, A., Magarini, M., Dev, K. and Dobre, O.A., 2025. Quantum Communication and Information Technologies: A Survey on Foundation, Error Correction, NISQ, and Networks. Authorea Preprints. Refer specifically to Section titled "A. Quantum Channel Models".
*/

class depolarizing_quantum_channel : public quantum_channel
{

private:
    // This is the Quantum Bit Error Rate which is equivalent to the probability of substitution to flip the bits (e.g., 0.06 for 6%)
    double qber;

// protected:
public:

    // Default constructor
    depolarizing_quantum_channel() : qber(0.0) {}

    void seedfrom(libbase::random& r) override
    {} // Does nothing.

    // The  state passes through the transmit method then noise is added at measurement.
    void transmit(computational_observable& observable) override
    {
        observable.set_qber(qber);
    }

    void transmit(hadamard_observable& observable) override
    {
        observable.set_qber(qber);
    }

     /*! \name Parameter handling */
    //! Set the characteristic parameters
    void set_parameters(const libbase::vector<double>& x) override
    {
        assertalways(x.size() ==
                     1); // Ensures all required parameters are passed
        qber = x(0);       // QBER (also known as the probability of substitution)
        assertalways(std::isfinite(qber) && qber >= 0.0);
    }

    //! Get the characteristic parameters
    libbase::vector<double> get_parameters() const override
    {
        libbase::vector<double> params;
        params.init(1);
        params(0) = qber;
        return params;
    }

    int get_num_params() const override
    {
        return 1;
    } // returns the number of CLI parameters
    // @}

    double get_qber() const override
    {
        return qber;
    }

    // Description - Returns a short string describing the channel
    std::string description() const override;

    //! Required for serializer registration - helper function
    // Helper function only used for testing.
    static std::unique_ptr<libbase::serializable> create(std::istream& sin)
    {
        auto obj = std::make_unique<depolarizing_quantum_channel>();
        obj->serialize(sin);
        return obj;
    }

    // Serialization Support
    DECLARE_SERIALIZER(depolarizing_quantum_channel)

};

} // namespace libcomm

#endif // __depolarizing_quantum_channel_h