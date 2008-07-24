/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "pacifier.h"
#include <sstream>

namespace libbase {

// Static interface

bool pacifier::quiet = false;

// Constructors / Destructors

pacifier::pacifier(const std::string& name) :
   name(name), t(name+"Pacifier"), last(0), characters(0)
   {
   // we don't want to start the timer until something is happening
   t.stop();
   }

pacifier::~pacifier()
   {
   // make sure we don't get spurious timer output
   t.stop();
   }

// Pacifier operation

/*! \brief Pacifier output
   Returns a string according to a input values specifying amount of work done.
   This function keeps a timer that automatically resets and stops (at
   beginning and end values respectively), to display estimated time remaining.
*/
std::string pacifier::update(int complete, int total)
   {
   // if output is disabled, return immediately
   if(quiet)
      return "";
   const int value = int(100*complete/double(total));
   // if we detect that we've started from zero again,
   // reset the timer and don't print anything
   if(complete == 0 || value < last)
      {
      t.start();
      last = value;
      characters = 0;
      return "";
      }
   // if we're at the last step, stop the timer
   // and return enough spaces to overwrite the last output
   if(complete == total)
      {
      t.stop();
      std::string s;
      if(characters > 0)
         {
         s.assign(characters,' ');
         s += '\r';
         }
      return s;
      }
   // otherwise we know we're someway in between...
   // estimate how long this whole stage will take to complete
   const double estimate = t.elapsed()/double(complete)*total;
   // return a blank if there is no change or if this won't take long enough
   if(value == last || estimate < 60)
      return "";
   // create the required string
   std::ostringstream sout;
   // if this is the first time we're printing something, start on a new line
   if(characters == 0)
      sout << '\n';
   sout << name << ": completed " << value << "%, elapsed " << t
        << " of estimated " << timer::format(estimate) << '\r';
   // update tracker
   last = value;
   characters = sout.str().length();
   return sout.str();
   }

}; // end namespace
