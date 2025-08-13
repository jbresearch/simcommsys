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

#include <memory>
#include <vector>


namespace libcomm {

class cvqkd_protocol : public qkd_protocol<double, libbase::vector>
{
    private:
         libbase::randgen rng; // used to randomly choose observables
         libbase::vector<int> decision_vector;
         libbase::vector<int> alice_decision_vector;

    public:
        void seedfrom(libbase::random& rng) override { this->rng.seed(rng.ival()); }

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

        // Description function
        std::string description() const override;

        libbase::vector<bool> postprocess(libbase::vector<double>&& alice_measurements,  libbase::vector<double>&& bob_measurements) override
        {
            failwith("postprocess() not yet implemented for cvqkd_protocol.");
            return libbase::vector<bool>();
        }

        DECLARE_SERIALIZER(cvqkd_protocol)
};


} // namespace libcomm


#endif // CVQKD_PROTOCOL_H