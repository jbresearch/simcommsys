#include "cvqkd_protocol.h"
#include <sstream>

using libbase::serializer;

namespace libcomm
{
    // Split fn to be used for parameter estimation and post-processing.
    std::tuple<libbase::vector<double>, // X_PE for Alice
    libbase::vector<double>, // Y_PE for Bob
    libbase::vector<double>, // Alice's raw key
    libbase::vector<double>> // Bob's raw key
    cvqkd_protocol::split(libbase::vector<double>& measurements_alice, libbase::vector<double>& measurements_bob,
    int N_PE)
    {
        assert(measurements_alice.size() == measurements_bob.size() && "Alice and Bob's measurement vector sizes are not equal.");

        assert(N_PE > 0 && "N_PE must be > 0.");

        const int N = measurements_alice.size();
        if (N_PE < 0) N_PE = 0;
        if (N_PE > N) N_PE = N;

        libbase::vector<double> X_PE, Y_PE, X_raw, Y_raw;
        X_PE.init(N_PE);
        Y_PE.init(N_PE);
        X_raw.init(N - N_PE);
        Y_raw.init(N - N_PE);

        // First N_PE -> PE
        for (int i = 0; i < N_PE; ++i) {
            X_PE(i) = measurements_alice(i);
            Y_PE(i)   = measurements_bob(i);
        }

        for (int i = N_PE; i < N; ++i) {
            const int j = i - N_PE;
            X_raw(j) = measurements_alice(i); // Unnormalised key of Alice to be used for post-processing
            Y_raw(j)   = measurements_bob(i); // Unnormalised key of Bob to be used for post-processing
        }

        return {X_PE, Y_PE, X_raw, Y_raw};
    }

    // Returns description of the protocol
    std::string cvqkd_protocol::description() const { return "CV-QKD Protocol using the GG02 protocol with GM Coherent states";}

    //! Serialize protocol
    std::ostream& cvqkd_protocol::serialize(std::ostream& sout) const
    {
        sout << "### Number of samples for Parameter Estimation N_PE" << std::endl;
        sout << N_PE << std::endl;
        return sout;
    }

    //! Deserialize protocol
    std::istream& cvqkd_protocol::serialize(std::istream& sin)
    {
        assertalways(sin.good());
        sin >> libbase::eatcomments >> N_PE >> libbase::verify;
        return sin;
    }

    const serializer cvqkd_protocol::shelper("qkd_protocol", "cvqkd_protocol", cvqkd_protocol::create);

} // namespace libcomm