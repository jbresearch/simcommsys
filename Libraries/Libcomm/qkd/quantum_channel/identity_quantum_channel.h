#ifndef __identity_quantum_channel_h
#define __identity_quantum_channel_h

#include "assertalways.h"
#include "qkd/quantum_channel.h"

namespace libcomm
{

/*!
 * \brief   Identity Quantum channel.
 * \author  Mark Mizzi
 *
 * Supports every observable.
 * Does not affect any observable, but leaves them as they are.
 */
class identity_quantum_channel : public quantum_channel
{
public:
    //! \name Visitor interface methods for observables
    void transmit(position_observable&) override {}
    void transmit(momentum_observable&) override {}
    void transmit(fake_position_observable&) override {}
    void transmit(fake_momentum_observable&) override {}
    void transmit(computational_observable&) override {}
    void transmit(hadamard_observable&) override {}
    void transmit(fake_computational_observable&) override {}
    void transmit(fake_hadamard_observable&) override {}
    //! @}

    void seedfrom(libbase::random& r) override {}

    /*! \name Parameter handling */
    //! Set the characteristic parameters
    void set_parameters(const libbase::vector<double>& x) override {}
    //! Get the characteristic parameters
    libbase::vector<double> get_parameters() const override
    {
        libbase::vector<double> params;
        params.init(0);
        return params;
    }
    int get_num_params() const override { return 0; }
    // @}

    // Description
    std::string description() const override;

    // Serialization Support
    DECLARE_SERIALIZER(identity_quantum_channel)
};

} // namespace libcomm

#endif // __identity_quantum_channel_h