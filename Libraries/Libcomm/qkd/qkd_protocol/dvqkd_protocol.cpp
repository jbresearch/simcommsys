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

    /* TODO: Still to implement and check below! */

    libbase::vector<bool> final_secret_key_KA;
    libbase::vector<bool> final_secret_key_KB;

    // Set a default length of 0. This is updated upon success.
    len_secret_key = 0;

    // Calculating N_PE: the number of samples used for parameter estimation.
    // N_PE = N (number of generated states) - n (size of codeword of the
    // codec)
    N_PE = alice_measurements.size() - get_codec_output_bits_n();

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

    return sin;
}

const serializer dvqkd_protocol::shelper("qkd_protocol",
                                         "dvqkd_protocol",
                                         dvqkd_protocol::create);



} // namespace libcomm