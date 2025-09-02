/*!
 * \brief CV-QKD Protocol
 * \author Aaron Abela
 *
 * Implements the steps that are exclusive to the CV-QKD protocol  namely the GG02 protocol with GM coherent states.
 *
 */

#ifndef CVQKD_PROTOCOL_H
#define CVQKD_PROTOCOL_H

#include "qkd/qkd_protocol.h"
#include "qkd/observable/position_observable.h"
#include "qkd/observable/momentum_observable.h"
#include "qkd/observable/fake_position_observable.h"
#include "qkd/observable/fake_momentum_observable.h"
#include "random.h"
#include "serializer.h"
#include "codec.h"

#include <memory>
#include <vector>


namespace libcomm {

class cvqkd_protocol : public qkd_protocol<double, libbase::vector>
{
    private:
        //  std::shared_ptr<codec<libbase::vector>> cdc; // error-correcting code
         libbase::randgen rng; // used to randomly choose observables
         libbase::vector<int> decision_vector;
         libbase::vector<int> alice_decision_vector;
         int N_PE; // Number of samples used for parameter estimation.
         int N_0; // shot noise
         double v_el; // electric noise
         double detector_efficiency;
         double smoothing_parameter;
         double I_AB; // Mutual Information between Alice and Bob.
         double chi_BE; // Holevo Bound between Bob and Eve for RR.
         int n_samples; //Number of samples after parameter estimation.
         double beta_mdr; // Reconciliation Efficiency for MDR.
         double SNR_linear; // Retrieved from bob's quantum channel.

    public:
        // void seedfrom(libbase::random& rng) override { this->rng.seed(rng.ival());  if (cdc) cdc->seedfrom(rng);}
        void seedfrom(libbase::random& rng) override { this->rng.seed(rng.ival());}

        // Note: here I replaced libbase::vector with the std::vector only for the observables.
        // Returns the observables of Bob
        std::vector<std::unique_ptr<observable<double>>> get_bob_observables(int framesize) override
        {
            std::vector<std::unique_ptr<observable<double>>> observables;
            observables.reserve(framesize);

            decision_vector.init(framesize);

            for (int i = 0; i < framesize; ++i) {
                if (rng.ival(2)==0){
                    observables.push_back(std::make_unique<position_observable>());
                    decision_vector(i) = 0;
                }
                else{
                    observables.push_back(std::make_unique<momentum_observable>());
                    decision_vector(i) = 1;
                }
            }

            return observables;
        }

        // Returns the observables of Alice
        std::vector<std::unique_ptr<observable<double>>> get_alice_observables(int framesize) override
        {
            std::vector<std::unique_ptr<observable<double>>> observables;
            observables.reserve(framesize);

            alice_decision_vector.init(framesize);

            for (int i = 0; i < framesize; ++i) {
                if (rng.ival(2)==0){
                    observables.push_back(std::make_unique<fake_position_observable>());
                    // Dummy test to check what alice created. To delete.
                    alice_decision_vector(i) = 0;
                }
                else{
                    observables.push_back(std::make_unique<fake_momentum_observable>());
                    // Dummy test to check what alice created. To delete.
                    alice_decision_vector(i) = 1;
                }
            }

            return observables;
        }

        // Also returns the observables of Alice, but this method also accepts Bob's decision vector
        std::vector<std::unique_ptr<observable<double>>> get_alice_observables(int framesize, const libbase::vector<int>& bobs_decision_vector) override
        {
            std::vector<std::unique_ptr<observable<double>>> observables;
            observables.reserve(framesize);

            alice_decision_vector.init(framesize);

            for (int i = 0; i < framesize; ++i) {
                if (bobs_decision_vector(i)==0){
                    observables.push_back(std::make_unique<fake_position_observable>());
                    alice_decision_vector(i) = 0;
                }
                else{

                    observables.push_back(std::make_unique<fake_momentum_observable>());
                    alice_decision_vector(i) = 1;
                }
            }

            return observables;
        }

        // Getter to access the decision vector to send to Alice. To delete, created just for testing.
        const libbase::vector<int>& get_alice_decision_vector() const {return alice_decision_vector;}

        // Getter fn to get Bob's decision vector to be used by Alice
        const libbase::vector<int>& get_decision_vector() const {return decision_vector;}

        // Getter to return various parameters for parameter estimation.
        int get_N_PE() override {return N_PE;}
        int get_N_0() override {return N_0;}
        double get_v_el() override {return v_el;}
        double get_det_eff() override {return detector_efficiency;}

        // Split fn to be used for parameter estimation and post-processing which returns: X_PE, Y_PE, X_raw and Y_raw
        std::tuple<libbase::vector<double>,
        libbase::vector<double>,
        libbase::vector<double>,
        libbase::vector<double>> split(libbase::vector<double>& measurements_alice, libbase::vector<double>& measurements_bob,
        int N_PE) override;

        // Parameter Estimation for the GG02 protocol using Optical Fiber which returns: T_hat, Epsilon_hat, chi_total_hat
        std::tuple<double, double, double> parameter_estimation_optical_fiber(libbase::vector<double>& X_PE, libbase::vector<double>& Y_PE, int N_0, double v_el, double detector_efficiency) override;

        // Mutual Information for the GG02 protocol
        double calculate_mutual_information(double chi_total_hat, double VA) override;

        // Helper functions used to calculate the Holevo Bound.
        // G(x) from Eq. (2.54). sTILL TO ADD REFERENCE
        inline double bosonic_entropy_G(double x) {
            if (x <= 0.0) return 0.0;
            return (x + 1.0) * std::log2(x + 1.0) - x * std::log2(x);
        }

        // Safe sqrt: clamp tiny negative values due to round-off
        inline double safe_sqrt(double x) {
            return std::sqrt(x < 0.0 ? 0.0 : x);
        }

        // Holevo Bound calculation for the GG02 protocol.
        double calculate_holevo_bound(double V, double T_hat, double Epsilon_hat, double X_total_hat) override;

        // Calculates the L2 norm.
        static double l2(const libbase::vector<double>& v) {
            long double s = 0.0L;
            for (int i = 0; i < v.size(); ++i) s += (long double)v(i) * v(i);
            return std::sqrt((double)s);
        }

        /* STILL TO ADD: Link int k to the codec I will be using e.g. LDPC */

        // Method to generate the bool vector s randomly generated by Bob.
        libbase::vector<bool> create_vector_s(libbase::random& rng, int k)
        {
            // k is the size of the vector that will be encoded to size n using LDPC
            libbase::vector<bool> s(k);
            for (int i = 0; i < k; ++i)
                s(i) = (rng.ival(2) != 0);   // 0 or 1 with equal probability
            return s;
        }

        double compute_beta_mdr(double code_rate, double snr_linear);

        void set_parameters_secret_key_length(double beta, double I_AB, double chi_BE, int n_samples) override
        {
            this->beta_mdr   = beta;
            this->I_AB       = I_AB;
            this->chi_BE     = chi_BE;
            this->n_samples  = n_samples;
        }

        const int calculate_finite_size_effects_secret_key_length();

        // Returns final secret key.
        libbase::vector<bool> postprocess(libbase::vector<double>&& alice_measurements,  libbase::vector<double>&& bob_measurements) override;

        // Description function
        std::string description() const override;

        DECLARE_SERIALIZER(cvqkd_protocol)
};




} // namespace libcomm


#endif // CVQKD_PROTOCOL_H