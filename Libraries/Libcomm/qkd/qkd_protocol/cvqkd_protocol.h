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

    public:
        void seedfrom(libbase::random& rng) override { this->rng.seed(rng.ival()); }

        // Note: here I replaced libbase::vector with the std::vector only for the observables.
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

        // Getter to access the decision vector to send to Alice
        const libbase::vector<int>& get_decision_vector() const {return decision_vector;}

        // Description function
        std::string description() const;

        // Still to implement the get_alice_observables and postprocess fns and here I replaced libbase::vector with the std::vector only for the observables.
        std::vector<std::unique_ptr<observable<double>>> get_alice_observables(int) override
        {
            failwith("get_alice_observables not implemented yet.");
            return {};
        }

        libbase::vector<bool> postprocess(libbase::vector<double>&& alice_measurements,  libbase::vector<double>&& bob_measurements) override
        {
            failwith("postprocess() not yet implemented for cvqkd_protocol.");
            return libbase::vector<bool>();
        }

        DECLARE_SERIALIZER(cvqkd_protocol)
};


} // namespace libcomm


#endif // CVQKD_PROTOCOL_H