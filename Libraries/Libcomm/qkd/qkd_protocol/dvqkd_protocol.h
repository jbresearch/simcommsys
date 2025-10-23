/*!
 * \file
 *
 * Copyright (c) 2025 Aaron Abela
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

/*!
 * \brief DV-QKD Protocol
 * \author Aaron Abela
 *
 * Implements the steps that are exclusive to the DV-QKD protocol namely the
 * BB84 protocol with single polarisation.
 *
 */

#ifndef DVQKD_PROTOCOL_H
#define DVQKD_PROTOCOL_H

#include "commsys.h"
#include "channel.h"
#include "crc/crc32.h"
#include "gf.h"
#include "hamming.h"
#include "qkd_commsys.h"
#include "qkd/observable/fake_hadamard_observable.h"
#include "qkd/observable/fake_computational_observable.h"
#include "qkd/observable/hadamard_observable.h"
#include "qkd/observable/computational_observable.h"
#include "qkd/privacy_amplification.h"
#include "qkd/privacy_amplification/pa_standard_toeplitz.h"
#include "qkd/qkd_protocol.h"
#include "qkd/quantum_state.h"
#include "random.h"
#include "serializer.h"
#include "source/quantum_bb84_source.h"

#include <memory>
#include <vector>


namespace libcomm
{

class dvqkd_protocol : public qkd_protocol<qubit, bool, libbase::vector>
{
private:
    libbase::randgen rng; // used to randomly choose observables
    libbase::vector<bool> bob_basis_vector; // vector b'
    libbase::vector<bool> alice_basis_vector; // vector b
    libbase::vector<bool> alice_bit_vector; // vector a

    std::shared_ptr<quantum_channel> m_bob_channel;

    // ADD THIS: Pointer to the source sequence from qkd_commsys fullcycle
    const libbase::vector<qubit>* m_source_sequence = nullptr;

    int N_PE;  // Number of samples used for parameter estimation.

    bool MI_Check = false;  // MI check that verifies if I_AB > X_B?
    bool H_check = false;    // Hash check that verifies if hash_hs == hash_hsat?
    int len_secret_key = 0; // Length of final secret key

protected:
    std::shared_ptr<codec<libbase::vector>> cdc; //!< Error-control codec

    //  Privacy Amplification System using the standard Toeplitz matrix
    pa_standard_toeplitz<bool> pa_system;

public:

    // Init method for DV-QKD protocol (BB84 protocol)
    void init(qkd_commsys<qubit, bool, libbase::vector>* qkdcommsys) override;

    // Description function
    std::string description() const override;

    void seedfrom(libbase::random& rng) override
    {
        this->rng.seed(rng.ival());
        if (cdc)
            cdc->seedfrom(rng);
        pa_system.seedfrom(rng);
    }

    void set_source_sequence(const libbase::vector<qubit>& source_sequence) override;

    std::pair<bool, bool> get_alice_choice_from_qubit(const libcomm::qubit& q);

    // Note: here I replaced libbase::vector with the std::vector only for the observables. Returns the observables of Bob
    std::vector<std::unique_ptr<observable<bool>>>
    get_bob_observables(int framesize) override
    {
        std::vector<std::unique_ptr<observable<bool>>> observables;
        observables.reserve(framesize);

        bob_basis_vector.init(framesize);

        for (int i = 0; i < framesize; ++i) {
            if (rng.ival(2) == 0) {
                observables.push_back(std::make_unique<computational_observable>());
                bob_basis_vector(i) = 0;
            } else {
                observables.push_back(std::make_unique<hadamard_observable>());
                bob_basis_vector(i) = 1;
            }
        }

        std::cout << "Printing basis vector b' of Bob (from dvqkd_protocol.h): " << std::endl;

        std::cout << "If b'(i) = 0 it is a Computational observable otherwise it is a Hadamard observable(from dvqkd_protocol.h): " << std::endl;

        std::cout << bob_basis_vector << std::endl;
        return observables;
    }

    // Returns the observables of Alice
    std::vector<std::unique_ptr<observable<bool>>>
    get_alice_observables(int framesize) override
    {
        std::vector<std::unique_ptr<observable<bool>>> observables;
        observables.reserve(framesize);

        alice_basis_vector.init(framesize); // vector b (basis)
        alice_bit_vector.init(framesize);   // vector a (bit)

        // Ensure the source sequence has been set by set_source_sequence()
        assert(m_source_sequence && "Source sequence was not set in dvqkd_protocol");
        assert(m_source_sequence->size() == framesize && "Source sequence size mismatch");

        // --- THIS IS THE NEW LOGIC ---
        for (int i = 0; i < framesize; ++i) {

            // Get the qubit from the stored sequence
            const qubit& q = (*m_source_sequence)(i);

            // Reverse-engineer the bit and basis from the qubit state
            std::pair<bool, bool> alice_choice = get_alice_choice_from_qubit(q);

            bool bit   = alice_choice.first;
            bool basis = alice_choice.second;

            // Store them in the protocol's member vectors
            alice_bit_vector(i)   = bit;   // This is Alice's bit vector a
            alice_basis_vector(i) = basis; // This is Alice's basis vector b

            // Create the corresponding fake observable for Alice
            if (basis == 0) { // Z-basis (Computational)
                observables.push_back(
                    std::make_unique<fake_computational_observable>());
            } else { // X-basis (Hadamard)
                observables.push_back(
                    std::make_unique<fake_hadamard_observable>());
            }
        }

        std::cout << "Printing basis vector b of Alice (from dvqkd_protocol.h): " << std::endl;

        std::cout << "If b(i) = 0 it is a fake Computational observable otherwise it is a fake Hadamard observable(from dvqkd_protocol.h): " << std::endl;

        std::cout << bob_basis_vector << std::endl;

        return observables;
    }

    // Split fn to be used for parameter estimation and post-processing which
    // returns: X_PE, Y_PE, X_raw and Y_raw
    std::tuple<libbase::vector<bool>,
               libbase::vector<bool>,
               libbase::vector<bool>,
               libbase::vector<bool>>
    split(libbase::vector<bool>& measurements_alice,
          libbase::vector<bool>& measurements_bob) override;


    // Helper functions related to the codec.
    int get_codec_input_bits_k() const override
    {
        return cdc->input_block_size();
    }

    int get_codec_output_bits_n() const override
    {
        return cdc->output_block_size();
    }

    // Helper function to Get codec.
    std::shared_ptr<codec<libbase::vector>> get_codec() const { return cdc; }

     // Returns final secret keys KA and KB.
    std::pair<libbase::vector<bool>, libbase::vector<bool>>
    postprocess(libbase::vector<bool>&& alice_measurements,
                libbase::vector<bool>&& bob_measurements) override;


    DECLARE_SERIALIZER(dvqkd_protocol)

};

} // namespace libcomm


#endif // DVQKD_PROTOCOL_H