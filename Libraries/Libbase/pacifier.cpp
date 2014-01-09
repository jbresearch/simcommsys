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

#include "pacifier.h"
#include <sstream>

namespace libbase {

// Static interface

bool pacifier::quiet = false;

// Constructors / Destructors

pacifier::pacifier(const std::string& name) :
   name(name), t(name + "Pacifier", false), last(0), characters(0)
   {
   }

// Pacifier operation

/*! \brief Pacifier output
 * Returns a string according to a input values specifying amount of work done.
 * This function keeps a timer that automatically resets and stops (at
 * beginning and end values respectively), to display estimated time remaining.
 */
std::string pacifier::update(int complete, int total)
   {
   // if output is disabled or not needed, return immediately
   if (quiet || total == 0)
      return "";
   const int value = int(100 * complete / double(total));
   // if we detect that we've started from zero again,
   // reset the timer and don't print anything
   if (complete == 0 || value < last)
      {
      t.start();
      last = value;
      characters = 0;
      return "";
      }
   // if we're at the last step, stop the timer
   // and return enough spaces to overwrite the last output
   if (complete == total)
      {
      t.stop();
      std::string s;
      if (characters > 0)
         {
         s.assign(characters, ' ');
         s += '\r';
         }
      return s;
      }
   // otherwise we know we're someway in between...
   // estimate how long this whole stage will take to complete
   const double estimate = t.elapsed() / double(complete) * total;
   // return a blank if there is no change or if this won't take long enough
   if (value == last || estimate < 60)
      return "";
   // create the required string
   std::ostringstream sout;
   // if this is the first time we're printing something, start on a new line
   if (characters == 0)
      sout << std::endl;
   sout << name << ": completed " << value << "%, elapsed " << t
         << " of estimated " << timer::format(estimate) << '\r';
   // update tracker
   last = value;
   characters = sout.str().length();
   return sout.str();
   }

} // end namespace
