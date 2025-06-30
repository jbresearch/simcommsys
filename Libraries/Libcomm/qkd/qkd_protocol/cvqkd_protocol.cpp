#include "cvqkd_protocol.h"
#include <sstream>

using libbase::serializer;

namespace libcomm
{
    // To uncomment once all issues of cvqkd_protocol.h are fixed
    // Returns description of the protocol
    std::string cvqkd_protocol::description() const { return "CV-QKD Protocol using the GG02 protocol with GM Coherent states";}

    //! Serialize protocol
    std::ostream& cvqkd_protocol::serialize(std::ostream& sout) const {
        return sout;
    }

    //! Deserialize protocol
    std::istream& cvqkd_protocol::serialize(std::istream& sin) {
        return sin;
    }

    const serializer cvqkd_protocol::shelper("qkd_protocol", "cvqkd_protocol", cvqkd_protocol::create);

} // namespace libcomm