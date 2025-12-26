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
    sout << "# Noise mean" << std::endl;
    sout << mean << std::endl;
    sout << "# Fading Coefficient alpha" << std::endl;
    sout << alpha << std::endl;
    return sout;
}

//! Deserialize channel from input stream
std::istream&
gaussian_quantum_channel::serialize(std::istream& sin)
{
    assertalways(sin.good());
    sin >> libbase::eatcomments >> mean >> libbase::verify;
    sin >> libbase::eatcomments >> alpha >> libbase::verify;
    return sin;
}

//! Register with serializer system
const libbase::serializer
    gaussian_quantum_channel::shelper("quantum_channel",
                                      "gaussian_quantum_channel",
                                      gaussian_quantum_channel::create);
} // namespace libcomm
