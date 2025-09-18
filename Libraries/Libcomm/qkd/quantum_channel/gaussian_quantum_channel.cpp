#include "qkd/quantum_channel/gaussian_quantum_channel.h"
#include "assertalways.h"
#include "serializer.h"

namespace libcomm
{

//! Return a string describing the channel
std::string
gaussian_quantum_channel::description() const
{
    return "gaussian_quantum_channel";
}

//! Serialize channel to output stream
std::ostream&
gaussian_quantum_channel::serialize(std::ostream& sout) const
{
    sout << "# Homodyne Detector Efficiency" << std::endl;
    sout << noise_detector_eff << std::endl;
    sout << "# Mean of the Gaussian Quantum Channel" << std::endl;
    sout << noise_mean << std::endl;
    sout << "# Transmittance T of the Gaussian Quantum Channel" << std::endl;
    sout << noise_transmittance << std::endl;
    return sout;
}

//! Deserialize channel from input stream
std::istream&
gaussian_quantum_channel::serialize(std::istream& sin)
{

    // assertalways(sin.good());
    sin >> libbase::eatcomments >> noise_detector_eff >>
        libbase::verify; // Detector Efficiency of the homodyne detector
    sin >> libbase::eatcomments >> noise_mean >>
        libbase::verify; // Mean of Noise
    sin >> libbase::eatcomments >> noise_transmittance >>
        libbase::verify; // Transmittance T
    return sin;
}

//! Register with serializer system
const libbase::serializer
    gaussian_quantum_channel::shelper("quantum_channel",
                                      "gaussian_quantum_channel",
                                      gaussian_quantum_channel::create);
} // namespace libcomm
