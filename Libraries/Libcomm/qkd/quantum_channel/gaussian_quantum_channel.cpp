#include "qkd/quantum_channel/gaussian_quantum_channel.h"
#include "serializer.h"
#include "assertalways.h"

using libbase::serializer;

namespace libcomm {

//! Return a string describing the channel
std::string gaussian_quantum_channel::description() const {
    return "gaussian_quantum_channel";
}

//! Serialize channel to output stream
std::ostream& gaussian_quantum_channel::serialize(std::ostream& sout) const {
    // sout << "Mean of the Gaussian Quantum Channel" << std::endl;
    // sout << noise_mean << std::endl;
    // sout << "Transmittance of the Gaussian Quantum Channel" << std::endl;
    // sout << noise_transmittance << std::endl;
    // sout << "Homodyne Detector Efficiency" << std::endl;
    // sout << noise_detector_eff << std::endl;
    return sout;
}

//! Deserialize channel from input stream
std::istream& gaussian_quantum_channel::serialize(std::istream& sin) {

    // assertalways(sin.good());
    // sin >> libbase::eatcomments >> noise_mean;
    // sin >> libbase::eatcomments >> noise_transmittance;
    // sin >> libbase::eatcomments >> noise_detector_eff;
    return sin;
}

//! Register with serializer system
const serializer gaussian_quantum_channel::shelper(
    "quantum_channel", "gaussian_quantum_channel", gaussian_quantum_channel::create);


}