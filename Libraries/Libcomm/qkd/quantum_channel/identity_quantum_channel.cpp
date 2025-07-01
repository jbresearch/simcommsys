/*!
 * \file identity_quantum_channel.cpp
 *
 * Copyright (c) 2025 Mark Mizzi
 *
 * This file is part of SimCommSys.
 * Released under the GNU General Public License v3 or later.
 */

#include "identity_quantum_channel.h"
#include "serializer.h"

using libbase::serializer;

namespace libcomm {

//! Return a string describing the channel
std::string identity_quantum_channel::description() const {
    return "identity_quantum_channel";
}

//! Serialize channel to output stream
std::ostream& identity_quantum_channel::serialize(std::ostream& sout) const {
    return sout;
}

//! Deserialize channel from input stream
std::istream& identity_quantum_channel::serialize(std::istream& sin) {
    return sin;
}

//! Register with serializer system
const serializer identity_quantum_channel::shelper(
    "quantum_channel", "identity_quantum_channel", identity_quantum_channel::create);

} // namespace libcomm


