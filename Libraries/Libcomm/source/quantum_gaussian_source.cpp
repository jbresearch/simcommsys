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


 #include "quantum_gaussian_source.h"
 #include <sstream>

 using libbase::serializer;

 namespace libcomm
 {

 // Register with serializer system
 const serializer quantum_gaussian_source::shelper(
 "source", "quantum_gaussian_source", quantum_gaussian_source::create);

  //! Description
 std::string
 quantum_gaussian_source::description() const {
     std::ostringstream sout;
     sout << "Quantum Gaussian Source ("
     << "q_mean ~ N(" << q_mean_mean << ", " << q_mean_stddev << "), "
     << "p_mean ~ N(" << p_mean_mean << ", " << p_mean_stddev << "))";
     return sout.str();
 }

 // Save mean and variance to stream
 std::ostream& quantum_gaussian_source::serialize(std::ostream& sout) const
 {
    return sout;
 }

 // Load mean and variance from stream
 std::istream& quantum_gaussian_source::serialize(std::istream& sin)
 {
    return sin;
 }

 } // namespace libcomm
