#include "qkd/quantum_channel/depolarizing_quantum_channel.h"
#include "assertalways.h"
#include "serializer.h"

namespace libcomm
{
    //! Return a string describing the channel
    std::string
    depolarizing_quantum_channel::description() const
    {
        return "depolarizing_quantum_channel";
    }

    //! Serialize channel to output stream
    std::ostream&
    depolarizing_quantum_channel::serialize(std::ostream& sout) const
    {
        return sout;
    }

    //! Deserialize channel from input stream
    std::istream&
    depolarizing_quantum_channel::serialize(std::istream& sin)
    {
        return sin;
    }

    //! Register with serializer system
    const libbase::serializer
        depolarizing_quantum_channel::shelper("quantum_channel", "depolarizing_quantum_channel",
                                        depolarizing_quantum_channel::create);

} // namespace libcomm