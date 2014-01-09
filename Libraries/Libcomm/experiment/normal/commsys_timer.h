/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
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

#ifndef __commsys_timer_h
#define __commsys_timer_h

#include "config.h"
#include "experiment/experiment_normal.h"
#include "experiment/binomial/commsys_simulator.h"
#include "experiment/binomial/result_collector/commsys/errors_hamming.h"

namespace libcomm {

/*!
 * \brief   Communication System Simulator - Timing collector.
 * \author  Johann Briffa
 *
 * A variation on the regular commsys_simulator object, returning component
 * timings as main result.
 */
template <class S>
class commsys_timer : public experiment_normal {
private:
   commsys_simulator<S, errors_hamming> simulator; //!< Base simulator object
   std::vector<double> timings; //!< List of timings from last cycle
   std::vector<std::string> names; //!< List of timer names from last cycle

public:
   // Experiment parameter handling
   void seedfrom(libbase::random& r)
      {
      simulator.seedfrom(r);
      }
   void set_parameter(const double x)
      {
      simulator.set_parameter(x);
      }
   double get_parameter() const
      {
      return simulator.get_parameter();
      }

   // Experiment handling
   void sample(libbase::vector<double>& result)
      {
      // Run the system simulation
      libbase::vector<double> temp;
      simulator.sample(temp);
      // Collect timings
      timings = simulator.get_timings();
      names = simulator.get_names();
      // Copy over timings as results
      result = libbase::vector<double>(timings);
      }
   int count() const
      {
      const size_t N = timings.size();
      assert(N == names.size());
      assert(N > 0);
      return N;
      }
   int get_multiplicity(int i) const
      {
      return 1;
      }
   std::string result_description(int i) const
      {
      assert(i >= 0 && i < int(names.size()));
      return names[i];
      }
   libbase::vector<int> get_event() const
      {
      return simulator.get_event();
      }

   // Description
   std::string description() const
      {
      return "Timed " + simulator.description();
      }

   // Serialization Support
DECLARE_SERIALIZER(commsys_timer)
};

} // end namespace

#endif
