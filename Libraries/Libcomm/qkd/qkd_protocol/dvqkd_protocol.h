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

#include "channel.h"
#include "channel/qsc.h"
#include "codec/codec_coset.h"
#include "codec/ldpc.h"
#include "modem.h"
#include "mapper.h"
#include "commsys.h"
#include "crc/crc32.h"
#include "gf.h"
#include "hamming.h"
#include "qkd/observable/computational_observable.h"
#include "qkd/observable/fake_computational_observable.h"
#include "qkd/observable/fake_hadamard_observable.h"
#include "qkd/observable/hadamard_observable.h"
#include "qkd/privacy_amplification.h"
#include "qkd/privacy_amplification/pa_standard_toeplitz.h"
#include "qkd/qkd_protocol.h"
#include "qkd/quantum_state.h"
#include "qkd_commsys.h"
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
    libbase::vector<bool> bob_basis_vector;   // vector b'
    libbase::vector<bool> alice_basis_vector; // vector b
    libbase::vector<bool> alice_bit_vector;   // vector a
    std::shared_ptr<quantum_channel> m_bob_channel;

    // ADD THIS: Pointer to the source sequence from qkd_commsys fullcycle
    const libbase::vector<qubit>* m_source_sequence = nullptr;

    int m_framesize; // Number of generated qubits for a single frame. 
    double Q_tol; // Maximum tolerated QBER
    double Q_worst_case; // 'Worst' case error rate
    int leak_EC; //  Bits revealed during error correction (syndrome length)
    int alphabet_size; // Used for demodulation and privacy amplification 

    bool H_check = false;   // Hash check that verifies if hash_hs == hash_hsat?
    int len_secret_key = 0; // Length of final secret key

    double QBER; // Estimate QBER between X_PE and Y_PE 
    bool estimate_parameters; //!< True for calculating QBER from parameter estimation. 
    double eps_sec; // Security parameter (e.g., 1e-10)
    double eps_cor; // Correctness parameter (e.g., 1e-15) 

protected:
    std::shared_ptr<codec_coset<libbase::vector>> cdc; //!< Error-control codec
    std::shared_ptr<blockmodem<libbase::gf2>> mdm; // modem
    std::shared_ptr<channel<libbase::gf2>> demodulation_channel;
    std::shared_ptr<mapper<libbase::vector>> map; // mapper 

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
        demodulation_channel->seedfrom(rng);
    }

    void
    set_source_sequence(const libbase::vector<qubit>& source_sequence) override;

    std::pair<bool, bool> get_alice_choice_from_qubit(const libcomm::qubit& q);

    // Note: here I replaced libbase::vector with the std::vector only for the
    // observables. Returns the observables of Bob
    std::vector<std::unique_ptr<observable<bool>>>
    get_bob_observables(int framesize) override;

    // Returns the observables of Alice
    std::vector<std::unique_ptr<observable<bool>>>
    get_alice_observables(int framesize) override;

    // Split fn to be used for parameter estimation and post-processing which
    // returns: X_PE, Y_PE, X_raw and Y_raw
    void 
    split(libbase::vector<bool>& alice_measurements,
          libbase::vector<bool>& bob_measurements, 
          libbase::vector<bool>& X, 
          libbase::vector<bool>& Y,
          libbase::vector<bool>& X_PE, 
          libbase::vector<bool>& Y_PE);
          
    double binary_entropy(double p);

    // Calculates the QBER and length l 
    std::pair<double, int>
    parameter_estimation(
    const libbase::vector<bool>& X_PE, const libbase::vector<bool>& Y_PE);

    const int calculate_finite_size_effects_secret_key_length() override;

    libbase::vector<int> pack_bits_to_symbols(const libbase::vector<bool>& bits, int m);

    template <class GFVec>
    void print_gf_vector_as_ints(const GFVec& v)
    {
        for (int i = 0; i < v.size(); ++i)
            std::cout << int(v(i)) << "\t";
        std::cout << std::endl;
    }

    // Returns final secret keys KA and KB.
    std::pair<libbase::vector<bool>, libbase::vector<bool>>
    postprocess(libbase::vector<bool>&& alice_measurements,
                libbase::vector<bool>&& bob_measurements) override;

    DECLARE_SERIALIZER(dvqkd_protocol)
};

} // namespace libcomm

#endif // DVQKD_PROTOCOL_H