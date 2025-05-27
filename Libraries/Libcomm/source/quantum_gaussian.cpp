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


 #include "quantum_gaussian.h"
 #include <sstream>

 namespace libcomm
 {

 // Save mean and variance to stream
 std::ostream& quantum_gaussian::serialize(std::ostream& sout) const
 {
     sout << "# Mean of Q_Mean" << std::endl;
     sout << q_mean_mean  << std::endl;
     sout << "# Stddev of Q_Mean" << std::endl;
     sout << q_mean_stddev << std::endl;
     sout << "# Mean of P_Mean" << std::endl;
     sout << p_mean_mean  << std::endl;
     sout << "# Stddev of P_Mean" << std::endl;
     sout << p_mean_stddev << std::endl;
     sout << "# Stddev of Q" << std::endl;
     sout << q_stddev << std::endl;
     sout << "# Stddev of P" << std::endl;
     sout << p_stddev << std::endl;

     return sout;
 }

 // Load mean and variance from stream
 std::istream& quantum_gaussian::serialize(std::istream& sin)
 {
     assertalways(sin.good());
     sin >> libbase::eatcomments >> q_mean_mean ;
     sin >> libbase::eatcomments >> q_mean_stddev;
     sin >> libbase::eatcomments >> p_mean_mean ;
     sin >> libbase::eatcomments >> p_mean_stddev;
     sin >> libbase::eatcomments >> q_stddev;
     sin >> libbase::eatcomments >> p_stddev;

     return sin;
 }

 } // namespace libcomm


 using libbase::serializer;

 namespace libcomm
 {

// Register with serializer system
const serializer quantum_gaussian::shelper(
    "source", "quantum_gaussian", quantum_gaussian::create);

 } // namespace libcomm
