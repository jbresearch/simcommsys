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
    std::mt19937 gen;

public:

    //! Constructor
    gaussian_quantum_channel()
        : noise_mean(0.0), noise_stddev(1.0), gen(){}

    //! Seeds the Mersenne Twister random number generator from a pseudo-random sequence
    void seedfrom(libbase::random& r) override {
        gen.seed(r.ival());
    }

    // Applies Gaussian noise to position observable
    void transmit(position_observable& observable) override {
        std::normal_distribution<double> q_dist(noise_mean, noise_stddev);
        double noise = q_dist(gen);
        observable.set_noise(noise);
    }

    // Applies Gaussian noise to momentum observable
    void transmit(momentum_observable& observable) override {
        std::normal_distribution<double> p_dist(noise_mean, noise_stddev);
        double noise = p_dist(gen);
        observable.set_noise(noise);
    }

    /*! \name Parameter handling */
    //! Set the characteristic parameters
    void set_parameters(const libbase::vector<double>& x) override {
        std::cout<< "Testing whether I am in the function" << std::endl;
        assertalways(x.size() == 1);  // Ensures all required parameters are passed
        noise_stddev = x(0); // Variance V_N
    }

    //! Get the characteristic parameters
    libbase::vector<double> get_parameters() const override
    {
        libbase::vector<double> params;
        params.init(1);
        params(0) = noise_stddev; // Variance V_N
        return params;
    }

    int get_num_params() const override { return 1; }
    // @}

    // Description - Returns a short string describing the channel
    std::string description() const override;

    // Serialization Support
    DECLARE_SERIALIZER(gaussian_quantum_channel)
};

}

#endif // __gaussian_quantum_channel_h