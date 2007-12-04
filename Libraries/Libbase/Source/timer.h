/* timer
 * ~~~~~
 * a class which can be used to time subroutines, etc. resolution is in microseconds,
 * but the class can also handle durations in weeks/years/anything you care to time...
 */

#ifndef __timer_h
#define __timer_h

#include "config.h"
#include "vcs.h"
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

/*
  Version 1.10 (6 Mar 1999)
  added a new function elapsed(), which returns the number of seconds elapsed.

  Version 1.11 (26 Apr 1999)
  added a new static function date(), which returns the current date and time.

  Version 1.12 (22 Jul 1999)
  added a new function usage(), which returns the percentage CPU usage.

  Version 1.13 (16 Apr 2001)
  stream output routine no longer stops the timer (to allow display of running timers).
  If anyone wants to stop a timer, this must be done explicitly.

  Version 1.20 (29 Sep 2001)
  added support for Win32, using the Performance Counter

  Version 1.21 (27 Oct 2001)
  added conversion function to generate a string, as an alternative output function.
  Also made the timer object a constant in the "<<" operator.

  Version 1.22 (31 Oct 2001)
  realised that the "<<" operator was messing up the stream's precision, so I fixed that.

  Version 1.23 (30 Nov 2001)
  modified operator char *() so that if time > 1min it will print seconds as integers.

  Version 1.24 (2 Dec 2001)
  removed stream << operator - stream output should be handled by operator char *().

  Version 1.30 (7 Feb 2002)
  modified the Win32 version to utilise a millisecond timer, and added a static time-
  formatting function. Also removed the (now unnecessary) tempstring.

  Version 1.31 (23 Feb 2002)
  modified the Win32 version to utilise the clock() timer to compute processor time.
  This should allow us finally to get a CPU usage estimate. Also added flushes to all
  end-of-line clog outputs, to clean up text user interface.

  Version 1.32 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.
  also changed use of iostream from global to std namespace.

  Version 1.33 (25 Mar 2002)
  changed the internal name variable from char* (which requires heap memory allocation
  and deallocation) to string; also changed the operator char* to operator string, and
  re-included a specific stream << output operator. The latter fixes a bug that otherwise
  stream output would merely show the address of the object - there is no automatic
  conversion. Finally, also changed the date() and format() functions to return string
  instead of a pointer to char. These changes in return type may require some modification
  to programs that use this class.
  Also changed the contructor parameter and the divide() parameter to const.
  Also removed the inline specifier from the declaration, and moved most functions into
  the implementation file - now only those functions which serve merely as a nomenclature
  are defined inline.

  Version 1.34 (27 Mar 2002)
  removed the divide() function - the same functionality can easily be obtained otherwise

  Version 1.35 (27 Apr 2002)
  fixed a bug in format() - when the elapsed time was >1 day, the number of days was
  still not shown - I was mistakenly overwriting the string with the HH:MM:SS rather
  than appending it. Fixed now.

  Version 1.36 (12 Jun 2002)
  included <stdio.h>, since this is needed for sprintf().

  Version 1.37 (15 Jun 2002)
  changed all pow() functions to more accurately specify that we want to
  call double pow(double, int)

  Version 1.38 (10 Oct 2006)
  updated class to work with VS .NET 2005 on Win32 platform:
  * added include for sys/types.h

  Version 1.40 (26 Oct 2006)
  * defined class and associated data within "libbase" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.
  
  Version 1.50 (23 Apr 2007)
  * renamed internal wallclock() and cputime() functions
  * added public function to obtain cputime used
  * refactored affected functions
*/

namespace libbase {

class timer {
   static const vcs version;
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
   timer(const char *n = NULL);
   virtual ~timer();
   void start();
   void stop();
   double elapsed() const;
   double cputime() const;
   double usage() const;

   operator std::string() const { return format(elapsed()); };

   friend std::ostream& operator<<(std::ostream& s, const timer& t) { return s << std::string(t); };

   static std::string date();
   static std::string format(const double elapsedtime);
};

}; // end namespace

#endif
