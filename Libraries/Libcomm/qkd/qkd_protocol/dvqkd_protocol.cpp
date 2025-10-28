#include "dvqkd_protocol.h"
#include "codec/ldpc.h"
#include <cmath>
#include <sstream>

using libbase::serializer;


namespace libcomm
{

// Determine debug level:
// 1 - Normal debug output only
#ifndef NDEBUG
#    undef DEBUG
#    define DEBUG 1
#endif

// Returns description of the protocol
std::string
dvqkd_protocol::description() const
{
    return "BB894 protocol with single polarization";
}

void
dvqkd_protocol::init(qkd_commsys<qubit, bool, libbase::vector>* qkdcommsys)
{
    // // Get the source generator from commsys
    // std::shared_ptr<source<gaussian_state>> src_gen_base = qkdcommsys->get_src();
    // assert(src_gen_base && "Commsys did not provide a source generator.");

    // // Safely cast to the derived class we need
    // auto& src_gen = dynamic_cast<quantum_bb84_source&>(*src_gen_base);

    // Get Bob's quantum channel from qkd_commsys
    this->m_bob_channel = qkdcommsys->get_bob_channel();
    assert(this->m_bob_channel && "qkd_commsys did not provide Bob's channel.");
}

void
dvqkd_protocol::set_source_sequence(const libbase::vector<qubit>& source_sequence)
{
    this->m_source_sequence = &source_sequence;
}

std::pair<bool, bool>
dvqkd_protocol::get_alice_choice_from_qubit(const libcomm::qubit& q)
{
    // Define the values Alice uses
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    const double epsilon = 1e-9; // A small tolerance for float comparison

    // Get the internal amplitudes.
    std::complex<double> alpha = q.get_comp_basis_0();
    std::complex<double> beta = q.get_comp_basis_1();

    // Based on the quantum_bb84_source.h, only the real parts need to be checked.

    // Case 1: State |0> (bit=0, basis=0)
    // alpha=1.0, beta=0.0
    if (std::abs(alpha.real() - 1.0) < epsilon && std::abs(beta.real()) < epsilon) {
        return {false, false}; // bit=0, basis=0 (Z)
    }

    // Case 2: State |1> (bit=1, basis=0)
    // alpha=0.0, beta=1.0
    if (std::abs(alpha.real()) < epsilon && std::abs(beta.real() - 1.0) < epsilon) {
        return {true, false}; // bit=1, basis=0 (Z)
    }

    // Case 3: State |+> (bit=0, basis=1)
    // alpha=1/sqrt(2), beta=1/sqrt(2)
    if (std::abs(alpha.real() - inv_sqrt2) < epsilon && std::abs(beta.real() - inv_sqrt2) < epsilon) {
        return {false, true}; // bit=0, basis=1 (X)
    }

    // Case 4: State |-> (bit=1, basis=1)
    // alpha=1/sqrt(2), beta=-1/sqrt(2)
    if (std::abs(alpha.real() - inv_sqrt2) < epsilon && std::abs(beta.real() + inv_sqrt2) < epsilon) {
        return {true, true}; // bit=1, basis=1 (X)
    }

    // If it's none of these, it's a state we don't recognize (e.g., noisy)
    throw std::runtime_error("Unknown qubit state. Not a valid Alice state.");
}


// Note: here I replaced libbase::vector with the std::vector only for the
    // observables. Returns the observables of Bob
std::vector<std::unique_ptr<observable<bool>>>
dvqkd_protocol::get_bob_observables(int framesize)
{
    std::vector<std::unique_ptr<observable<bool>>> observables;
    observables.reserve(framesize);

    bob_basis_vector.init(framesize);

    for (int i = 0; i < framesize; ++i) {
        if (rng.ival(2) == 0) {
            observables.push_back(
                std::make_unique<computational_observable>());
            bob_basis_vector(i) = 0;
        } else {
            observables.push_back(std::make_unique<hadamard_observable>());
            bob_basis_vector(i) = 1;
        }
    }

#if DEBUG >= 1
    std::cout << "DV_QKDPROTOCOL: Basis of Bob: " << bob_basis_vector << std::endl;
    // std::cout << "DV_QKDPROTOCOL: If b'(i) = 0 it is a Computational observable otherwise it is a Hadamard observable. "
                // << std::endl;
#endif

    std::cout << bob_basis_vector << std::endl;
    return observables;
}


    // Returns the observables of Alice
    std::vector<std::unique_ptr<observable<bool>>>
    dvqkd_protocol::get_alice_observables(int framesize)
    {
        std::vector<std::unique_ptr<observable<bool>>> observables;
        observables.reserve(framesize);

        alice_basis_vector.init(framesize); // vector b (basis)
        alice_bit_vector.init(framesize);   // vector a (bit)

        // Ensure the source sequence has been set by set_source_sequence()
        assert(m_source_sequence &&
               "Source sequence was not set in dvqkd_protocol");
        assert(m_source_sequence->size() == framesize &&
               "Source sequence size mismatch");

        for (int i = 0; i < framesize; ++i) {

            // Get the qubit from the stored sequence
            const qubit& q = (*m_source_sequence)(i);

            // Reverse-engineer the bit and basis from the qubit state
            std::pair<bool, bool> alice_choice = get_alice_choice_from_qubit(q);

            bool bit = alice_choice.first;
            bool basis = alice_choice.second;

            // Store them in the protocol's member vectors
            alice_bit_vector(i) = bit;     // This is Alice's bit vector a
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


#if DEBUG >= 1
    std::cout << "DV_QKDPROTOCOL: Basis of Alice: " << alice_basis_vector << std::endl;
    // std::cout << "DV_QKDPROTOCOL: If b(i) = 0 it is a fake Computational observable otherwise it is a fake Hadamard observable. "
    //             << std::endl;
#endif

        return observables;
    }



// Split fn to be used for parameter estimation and post-processing.
std::tuple<libbase::vector<bool>, // X_PE for Alice
           libbase::vector<bool>, // Y_PE for Bob
           libbase::vector<bool>, // Alice's raw key
           libbase::vector<bool>> // Bob's raw key
dvqkd_protocol::split(libbase::vector<bool>& alice_measurements,
                      libbase::vector<bool>& bob_measurements)
{
    assert(alice_measurements.size() == bob_measurements.size() &&
           "Alice and Bob's measurement vector sizes are not equal.");

    assert(N_PE > 0 && "N_PE must be > 0.");

    const int N = alice_measurements.size();

    libbase::vector<bool> X_PE, Y_PE, X_raw, Y_raw;
    X_PE.init(N_PE);
    Y_PE.init(N_PE);
    X_raw.init(N - N_PE);
    Y_raw.init(N - N_PE);

    // First N_PE -> PE
    for (int i = 0; i < N_PE; ++i) {
        X_PE(i) = alice_measurements(i);
        Y_PE(i) = bob_measurements(i);
    }

    for (int i = N_PE; i < N; ++i) {
        const int j = i - N_PE;
        X_raw(j) = alice_measurements(
            i); // Unnormalised key of Alice to be used for post-processing
        Y_raw(j) = bob_measurements(
            i); // Unnormalised key of Bob to be used for post-processing
    }

#if DEBUG >= 1
    std::cerr << "DV_QKDPROTOCOL: alice_measurements = " << alice_measurements
              << std::endl;
    std::cerr << "DV_QKDPROTOCOL: bob_measurements = " << bob_measurements
              << std::endl;
    std::cerr << "DV_QKDPROTOCOL: X_PE = " << X_PE << std::endl;
    std::cerr << "DV_QKDPROTOCOL: Y_PE = " << Y_PE << std::endl;
    std::cerr << "DV_QKDPROTOCOL: X_raw = " << X_raw << std::endl;
    std::cerr << "DV_QKDPROTOCOL: Y_raw = " << Y_raw << std::endl;
#endif

    return {X_PE, Y_PE, X_raw, Y_raw};
}

const int dvqkd_protocol::calculate_finite_size_effects_secret_key_length()
{
    const int secret_key_length = 0;

    /* Still to add and implement the equation that calculates the length of the final secret key. */

    return secret_key_length;
}


// Returns final secret keys KA and KB.
std::pair<libbase::vector<bool>, libbase::vector<bool>>
dvqkd_protocol::postprocess(libbase::vector<bool>&& alice_measurements,
                            libbase::vector<bool>&& bob_measurements)
{

    libbase::vector<bool> final_secret_key_KA;
    libbase::vector<bool> final_secret_key_KB;

    // Set a default length of 0. This is updated upon success.
    len_secret_key = 0;

    /* Sifting Step

    In this step we need to discard the bits where the basis vectors of Alice (vector b) and Bob (vector b') do not match.

    Step 1 - Get and store the indices of the elements of the basis vectors that won't match.
    */

    std::vector<int> diff_indices;

    if (bob_basis_vector.size() != alice_basis_vector.size()) {
        std::cerr << "Vectors are not the same size, so all indices beyond the smaller size will be different." << std::endl;
    } else {
        for (int i = 0; i < alice_basis_vector.size(); ++i) {
            // Get and store the indices of the elements of the basis vectors that won't match.
            if (alice_basis_vector(i) != bob_basis_vector(i)) {
                diff_indices.push_back(i);
            }
        }
    }

#if DEBUG >= 1
        std::cerr << "DV_QKDPROTOCOL: **** SIFTING PROCESS **** " << std::endl;
        std::cerr << "DV_QKDPROTOCOL: Indices of Mismatch in the elements of the basis vectors = " << diff_indices
                  << std::endl;
#endif

    if (alice_measurements.size() != bob_measurements.size()) {
        std::cerr << "Measurement vectors are not the same size." << std::endl;
    }

    // Delete the bits from both measurement vectors with those indices.
    const int n_original = alice_measurements.size();
    const int n_sifted = n_original - (int)diff_indices.size();

    libbase::vector<bool> sifted_alice_key;
    libbase::vector<bool> sifted_bob_key;
    sifted_alice_key.init(n_sifted);
    sifted_bob_key.init(n_sifted);

    int sifted_idx = 0;
    int diff_idx = 0;

    for (int i = 0; i < n_original; ++i) {
        // Check if the current index i is the next one to exclude.
        if (diff_idx < (int)diff_indices.size() && diff_indices[diff_idx] == i) {
            // Exclude index. Move to the next index in diff_indices.
            diff_idx++;
        } else {
            // Index's basis matched. Keep the measurement.
            sifted_alice_key(sifted_idx) = alice_measurements(i);
            sifted_bob_key(sifted_idx) = bob_measurements(i);
            sifted_idx++;
        }
    }

    assert(sifted_idx == n_sifted);

#if DEBUG >= 1
    std::cerr << "DV_QKDPROTOCOL: Original measurement vector size = " << n_original << std::endl;
    std::cerr << "DV_QKDPROTOCOL: Size of sifted keys = " << n_sifted << std::endl;
    std::cerr << "DV_QKDPROTOCOL: Sifted Alice Key = " << sifted_alice_key << std::endl;
    std::cerr << "DV_QKDPROTOCOL: Sifted Bob Key = " << sifted_bob_key << std::endl;
#endif


    // // Calculating N_PE: the number of samples used for parameter estimation.
    // // N_PE = N (number of generated states) - n (size of codeword of the
    // // codec)
    // N_PE = alice_measurements.size() - get_codec_output_bits_n();

#if DEBUG >= 1
        std::cerr << "DV_QKDPROTOCOL: len_secret_key = " << len_secret_key
                  << std::endl;
        std::cerr << "DV_QKDPROTOCOL: final_secret_key_KA = "
                  << final_secret_key_KA << std::endl;
        std::cerr << "DV_QKDPROTOCOL: final_secret_key_KB = "
                  << final_secret_key_KB << std::endl;
#endif

        // print final keys
        return {std::move(final_secret_key_KA), std::move(final_secret_key_KB)};
}


//! Serialize protocol
std::ostream&
dvqkd_protocol::serialize(std::ostream& sout) const
{
    sout << "# Codec" << std::endl;
    sout << cdc << std::endl;
    return sout;
}

//! Deserialize protocol
std::istream&
dvqkd_protocol::serialize(std::istream& sin)
{
    assertalways(sin.good());
    sin >> libbase::eatcomments >> cdc >> libbase::verify;

    // check that all assumptions hold
    assertalways(cdc->num_inputs() == 2); // input has to be binary
    assertalways(cdc->num_outputs() == 2); // output has to be binary

    return sin;
}

const serializer dvqkd_protocol::shelper("qkd_protocol",
                                         "dvqkd_protocol",
                                         dvqkd_protocol::create);



} // namespace libcomm