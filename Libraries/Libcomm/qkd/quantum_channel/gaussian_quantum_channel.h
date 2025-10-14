#ifndef __gaussian_quantum_channel_h
#define __gaussian_quantum_channel_h

#include "qkd/observable/momentum_observable.h"
#include "qkd/observable/position_observable.h"
#include "qkd/quantum_channel.h"
#include <cmath>
#include <random>

namespace libcomm
{

/*!
 * \brief   Gaussian Quantum channel.
 * \author  Aaron Abela
 *
 * Supports position and momentum observable.
 * Noise is modelled as a Normal Distribution with Mean zero and standard
 * deviation (Variance V_N).
 */

class gaussian_quantum_channel : public quantum_channel
{

private:
    double noise_transmittance; // Transmittance
    double noise_detector_eff;  // homodyne detector efficiency
    double SNR; // Only CLI parameter of the quantum channel (linear not dB)
    double VA; // Modulation variance of Alice that will be set by a qkd_commsys
               // object.
    double noise_mean; // Chosen mean of the ND to generate the noise. This is a
                       // serialized parameter.
    std::mt19937 gen;

protected:
    double compute_VN() // Calculates variance VN to generate the noisy coherent
                        // states.
    {
        assertalways(std::isfinite(VA) && VA >= 0.0);
        assertalways(std::isfinite(SNR) && SNR > 0.0);
        // std::cout << "\nPrinting VN from gaussian_quantum_channel.h\n = " <<
        // VA/SNR << std::endl;
        return VA / SNR;
    }

    template <class Obs>
    void transmit_impl(Obs& observable)
    {
        const double VN =
            compute_VN(); // VA has to be set before calling transmit fn
        std::cout << "Value of VN from gaussian_channel.h = " << VN <<
        std::endl;
        const double noise_stddev = std::sqrt(VN);
        std::normal_distribution<double> dist(noise_mean, noise_stddev);
        const double noise = dist(gen);
        observable.set_noise(noise);
        observable.set_transmittance(
            noise_transmittance);                        // Serialized parameter
        observable.set_detector_eff(noise_detector_eff); // Serialized parameter
    }

public:
    //! Constructor
    gaussian_quantum_channel()
        : noise_transmittance(0.0), noise_detector_eff(0.0), SNR(0.0), VA(0.0),
          noise_mean(0.0), gen()
    {
    }

    //! Seeds the Mersenne Twister random number generator from a pseudo-random
    //! sequence
    void seedfrom(libbase::random& r) override { gen.seed(r.ival()); }

    void set_VA(double va) override
    {
        assertalways(std::isfinite(va) && va >= 0.0);
        VA = va;
    }

    double get_VA() override { return VA; }

    // Applies Gaussian noise to momentum and position observable
    void transmit(position_observable& observable) override
    {
        transmit_impl(observable);
    }
    void transmit(momentum_observable& observable) override
    {
        transmit_impl(observable);
    }

    /*! \name Parameter handling */
    //! Set the characteristic parameters
    void set_parameters(const libbase::vector<double>& x) override
    {
        assertalways(x.size() ==
                     1); // Ensures all required parameters are passed
        SNR = x(0);      // SNR
        assertalways(std::isfinite(SNR) && SNR > 0.0);
    }

    //! Get the characteristic parameters
    libbase::vector<double> get_parameters() const override
    {
        libbase::vector<double> params;
        params.init(1);
        params(0) = SNR;
        return params;
    }

    int get_num_params() const override
    {
        return 1;
    } // returns the number of CLI parameters
    // @}

    // Description - Returns a short string describing the channel
    std::string description() const override;

    //! Required for serializer registration - helper function
    // Helper function only used for unit testing
    static std::unique_ptr<libbase::serializable> create(std::istream& sin)
    {
        auto obj = std::make_unique<libcomm::gaussian_quantum_channel>();
        obj->serialize(sin);
        return obj;
    }

    // Serialization Support
    DECLARE_SERIALIZER(gaussian_quantum_channel)
};

} // namespace libcomm

#endif // __gaussian_quantum_channel_h