#ifndef __gaussian_quantum_channel_h
#define __gaussian_quantum_channel_h

#include "qkd/quantum_channel.h"
#include "qkd/observable/position_observable.h"
#include "qkd/observable/momentum_observable.h"

namespace libcomm {

/*!
 * \brief   Gaussian Quantum channel.
 * \author  Aaron Abela
 *
 * Supports position and momentum observable.
 * Noise is modelled as a Normal Distribution with Mean zero and standard deviation (Variance V_N).
 * Note: Both the mean and standard deviations will be inputs by the user. The mean will be a serialized parameter whereas the standard deviation will be a CLI parameter.
 */

class gaussian_quantum_channel : public quantum_channel
{

private:
    double noise_mean; // Chosen mean of the ND to generate the noise. This is a serialized parameter.
    double noise_stddev; // Chosen stddev of the normal distribution to generate noise. This is a CLI parameter.
    double noise_transmittance; // Transmittance
    double noise_detector_eff; // homodyne detector efficiency
    std::mt19937 gen;

public:

    //! Constructor
    gaussian_quantum_channel()
        : noise_mean(0.0), noise_stddev(1.0), noise_transmittance(0.0), noise_detector_eff(0.0), gen(){}

    //! Seeds the Mersenne Twister random number generator from a pseudo-random sequence
    void seedfrom(libbase::random& r) override {
        gen.seed(r.ival());
    }

    // Applies Gaussian noise to position observable
    void transmit(position_observable& observable) override {
        std::normal_distribution<double> q_dist(noise_mean, noise_stddev);
        double noise = q_dist(gen);
        observable.set_noise(noise);
        observable.set_transmittance(noise_transmittance);
        std::cout<<"Printing detector efficiency in transmit fn: " << noise_detector_eff << std::endl;
        observable.set_detector_eff(noise_detector_eff);
    }

    // Applies Gaussian noise to momentum observable
    void transmit(momentum_observable& observable) override {
        std::normal_distribution<double> p_dist(noise_mean, noise_stddev);
        double noise = p_dist(gen);
        observable.set_noise(noise);
        observable.set_transmittance(noise_transmittance);
        std::cout<<"Printing detector efficiency in transmit fn: " << noise_detector_eff << std::endl;
        observable.set_detector_eff(noise_detector_eff);
    }

    /*! \name Parameter handling */
    //! Set the characteristic parameters
    void set_parameters(const libbase::vector<double>& x) override {
        assertalways(x.size() == 1);  // Ensures all required parameters are passed
        // noise_mean = x(0); // Mean of Noise
        noise_stddev = x(0); // Variance V_N
        // noise_transmittance = x(2); // Transmittance T
        // noise_detector_eff = x(3); // Homodyne Detector Efficiency
    }

    //! Get the characteristic parameters
    libbase::vector<double> get_parameters() const override
    {
        libbase::vector<double> params;
        params.init(1); // Order: stddev
        params(0) = noise_stddev; // Variance V_N (should be the only CLI parameter)
        return params;
    }

    int get_num_params() const override { return 1; } // returns the number of CLI parameters
    // @}

    // Description - Returns a short string describing the channel
    std::string description() const override;

    //! Required for serializer registration - helper function
    // Helper function only used for unit testing
    static std::unique_ptr<libbase::serializable> create(std::istream& sin) {
        auto obj = std::make_unique<libcomm::gaussian_quantum_channel>();
        obj->serialize(sin);
        return obj;
    }

    // Serialization Support
    DECLARE_SERIALIZER(gaussian_quantum_channel)
};

}

#endif // __gaussian_quantum_channel_h