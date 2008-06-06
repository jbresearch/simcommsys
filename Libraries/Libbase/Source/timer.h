#ifndef __timer_h
#define __timer_h

#include "config.h"
#include <time.h>
#ifdef WIN32
#  include <sys/types.h>
#  include <sys/timeb.h>
#else
#  include <sys/time.h>
#  include <sys/resource.h>
#endif
#include <string>
#include <iostream>

namespace libbase {

/*!
   \brief   Timer.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   A class which can be used to time subroutines, etc. resolution is in
   microseconds, but the class can also handle durations in
   weeks/years/anything you care to time...

   \note Win32 supported using a millisecond timer. The clock() timer
         is used to compute processor time. This allows us to get a CPU
         usage estimate.
*/

class timer {
   std::string name;
   double wall, cpu;
   bool running;
#ifdef WIN32
   double convert(const struct _timeb& tb) const { return((double)tb.time + (double)tb.millitm * 1E-3); };
#else
   double convert(const struct timeval& tv) const { return((double)tv.tv_sec + (double)tv.tv_usec * 1E-6); };
#endif
   double _wallclock() const;
   double _cputime() const;
public:
   explicit timer(const char *n = NULL);
   virtual ~timer();
   void start();
   void stop();
   //! The number of seconds elapsed.
   double elapsed() const;
   double cputime() const;
   //! The percentage CPU usage.
   double usage() const;

   //! Conversion function to generate a string
   operator std::string() const { return format(elapsed()); };

   //! The current date and time.
   static std::string date();
   static std::string format(const double elapsedtime);
};

/*!
   \brief Stream output
   
   \note This routine does not stop the timer, therefore allowing
         display of running timers.
*/
inline std::ostream& operator<<(std::ostream& s, const timer& t)
   {
   return s << std::string(t);
   };

}; // end namespace

#endif
