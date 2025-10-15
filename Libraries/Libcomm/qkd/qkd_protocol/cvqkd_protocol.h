/*!
 * \brief CV-QKD Protocol
 * \author Aaron Abela
 *
 * Implements the steps that are exclusive to the CV-QKD protocol  namely the
 * GG02 protocol with GM coherent states.
 *
 */

#ifndef CVQKD_PROTOCOL_H
#define CVQKD_PROTOCOL_H

#include "commsys.h"

#include "channel.h"
#include "channel/awgn1d.h"
#include "codec.h"
#include "crc/crc32.h"
#include "gf.h"
#include "hamming.h"
#include "informed_embedder/direct_block_informed_embedder.h"
#include "informed_embedder/sign.h"
#include "source/quantum_gaussian_source.h"
#include "qkd/quantum_state.h"
#include "qkd/observable/fake_momentum_observable.h"
#include "qkd/observable/fake_position_observable.h"
#include "qkd/observable/momentum_observable.h"
#include "qkd/observable/position_observable.h"
#include "qkd/privacy_amplification.h"
#include "qkd/privacy_amplification/pa_standard_toeplitz.h"
#include "qkd/qkd_protocol.h"
#include "random.h"
#include "serializer.h"

#include <memory>
#include <vector>

namespace libcomm
{

class cvqkd_protocol : public qkd_protocol<gaussian_state, double, libbase::vector>
{
private:
    libbase::randgen rng; // used to randomly choose observables
    libbase::vector<int> decision_vector;
    std::shared_ptr<quantum_channel> m_bob_channel;

    int framesize = 0; // Number of generated coherent states per frame. 
    int N_PE;    // Number of samples used for parameter estimation.
    int N_0;     // shot noise
    double v_el; // electric noise
    double detector_efficiency;
    double m_modulation_variance = 0.0;
    double smoothing_parameter;
    double I_AB = 0.0;      // Mutual Information between Alice and Bob.
    double chi_BE = 0.0;    // Holevo Bound between Bob and Eve for RR.
    bool MI_Check = false;  // Check that verifies if I_AB > X_B?
    int len_secret_key = 0; // Length of final secret key
    int n_samples = 0;      // Number of samples after parameter estimation.
                            // Equivalent to same n of LDPC codec.

    // Vector s from Bob from qkd_commsys
    libbase::vector<bool> bob_vector_s;

    // Vector C from Bob to be used only within the post-processing method.
    libbase::vector<bool> bob_vector_c;

    double beta_mdr;         // Reconciliation Efficiency for MDR.
    double SNR_linear = 0.0; // Retrieved from bob's quantum channel.

    // Alphabet size to be used in embedder for modem and privacy amplification.
    int alphabet_size;

protected:
    std::shared_ptr<codec<libbase::vector>> cdc; //!< Error-control codec
    std::shared_ptr<block_informed_embedder<double, libbase::vector, double>>
        embedder; // Embedder
    std::shared_ptr<channel<double>>
        demodulation_channel; // Channel to be used for demodulation.

    // Check that verifies if hash_hs == hash_hsat?
    int H_check = 0;

    //  Privacy Amplification System using the standard Toeplitz matrix
    pa_standard_toeplitz<bool> pa_system;

public:
    void seedfrom(libbase::random& rng) override
    {
        this->rng.seed(rng.ival());
        if (cdc)
            cdc->seedfrom(rng);
        pa_system.seedfrom(rng);
    }

    void initialise(source<gaussian_state>& src_gen_base) override;
    
    // Note: here I replaced libbase::vector with the std::vector only for the
    // observables. Returns the observables of Bob
    std::vector<std::unique_ptr<observable<double>>>
    get_bob_observables(int framesize) override
    {
        std::vector<std::unique_ptr<observable<double>>> observables;
        observables.reserve(framesize);

        decision_vector.init(framesize);

        for (int i = 0; i < framesize; ++i) {
            if (rng.ival(2) == 0) {
                observables.push_back(std::make_unique<position_observable>());
                decision_vector(i) = 0;
            } else {
                observables.push_back(std::make_unique<momentum_observable>());
                decision_vector(i) = 1;
            }
        }

        return observables;
    }

    // Returns the observables of Alice
    std::vector<std::unique_ptr<observable<double>>>
    get_alice_observables(int framesize) override
    {
        std::vector<std::unique_ptr<observable<double>>> observables;
        observables.reserve(framesize);

        for (int i = 0; i < framesize; ++i) {
            // Using Bob's decision_vector.
            if (decision_vector(i) == 0) {
                observables.push_back(
                    std::make_unique<fake_position_observable>());
            } else {
                observables.push_back(
                    std::make_unique<fake_momentum_observable>());
            }
        }

        return observables;
    }

    // Split fn to be used for parameter estimation and post-processing which
    // returns: X_PE, Y_PE, X_raw and Y_raw
    std::tuple<libbase::vector<double>,
               libbase::vector<double>,
               libbase::vector<double>,
               libbase::vector<double>>
    split(libbase::vector<double>& measurements_alice,
          libbase::vector<double>& measurements_bob) override;

    // Parameter Estimation for the GG02 protocol using Optical Fiber which
    // returns: T_hat, Epsilon_hat, chi_total_hat
    std::tuple<double, double, double>
    parameter_estimation_optical_fiber(const libbase::vector<double>& X_PE,
                                       const libbase::vector<double>& Y_PE);

    // Method that gets VA from Bob's quantum channel initialised in
    // qkd_commsys.h
    void prepare_for_cycle(const std::shared_ptr<quantum_channel>& bob_channel) override;

    // Mutual Information for the GG02 protocol.
    double calculate_mutual_information(double chi_total_hat);

    // Helper functions used to calculate the Holevo Bound.
    // G(x) from Eq. (2.54). sTILL TO ADD REFERENCE
    inline double bosonic_entropy_G(double x)
    {
        if (x <= 0.0)
            return 0.0;
        return (x + 1.0) * std::log2(x + 1.0) - x * std::log2(x);
    }

    // Safe sqrt: clamp tiny negative values due to round-off
    inline double safe_sqrt(double x) { return std::sqrt(x < 0.0 ? 0.0 : x); }

    // Holevo Bound calculation for the GG02 protocol.
    double calculate_holevo_bound(double T_hat,
                                  double Epsilon_hat,
                                  double X_total_hat);

    double calculate_shannon_capacity_awgn()
    { // bits/use to be used to compute Beta for MDR.
        return 0.5 * std::log2(1.0 + SNR_linear);
    }

    // Calculates the L2 norm.
    static double l2(const libbase::vector<double>& v)
    {
        long double s = 0.0L;
        for (int i = 0; i < v.size(); ++i)
            s += (long double)v(i) * v(i);
        return std::sqrt((double)s);
    }

    double compute_beta_mdr(double code_rate);

    // Equations related to length of final secret key.
    const int calculate_finite_size_effects_secret_key_length();

    // Helper functions related to the codec.
    int get_codec_input_bits_k() const override
    {
        return cdc->input_block_size();
    }

    int get_codec_output_bits_n() const override
    {
        return cdc->output_block_size();
    }

    // Getter to get Bob's vector s from qkd_commsys
    void set_bob_vector_s(libbase::vector<bool>& s) override
    {
        bob_vector_s = s;
    }

    // Helper function to Get codec.
    std::shared_ptr<codec<libbase::vector>> get_codec() const { return cdc; }

    // Returns final secret keys KA and KB.
    std::pair<libbase::vector<bool>, libbase::vector<bool>>
    postprocess(libbase::vector<double>&& alice_measurements,
                libbase::vector<double>&& bob_measurements) override;

    // Description function
    std::string description() const override;

    DECLARE_SERIALIZER(cvqkd_protocol)
};

} // namespace libcomm

#endif // CVQKD_PROTOCOL_H