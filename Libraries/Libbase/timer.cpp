/*!
 * \file
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

#include "timer.h"
#include <cmath>
#include <cstdio>
#include <cstring>

namespace libbase {

// system-dependent functions

#ifdef WIN32

double timer::_wallclock() const
   {
   struct _timeb tb;
   _ftime(&tb);
   return convert(tb);
   }

double timer::_cputime() const
   {
   return clock()/double(CLOCKS_PER_SEC);
   }

#else //(ifdef WIN32)
double timer::_wallclock() const
   {
   struct timeval tv;
   struct timezone tz;

   gettimeofday(&tv, &tz);

   return convert(tv);
   }

double timer::_cputime() const
   {
   struct rusage usage;
   double cpu;

   getrusage(RUSAGE_SELF, &usage);
   cpu = convert(usage.ru_utime);
   getrusage(RUSAGE_CHILDREN, &usage);
   cpu += convert(usage.ru_utime);

   return (cpu);
   }

#endif //(ifdef WIN32)
// common functions

timer::timer(const std::string& name)
   {
   timer::name = name;
   start();
   }

timer::~timer()
   {
   if (running)
      {
      std::clog << "Timer";
      if (name != "")
         std::clog << " (" << name << ")";
      std::clog << " expired after " << *this << std::endl;
      }
   }

void timer::start()
   {
   wall = _wallclock();
   cpu = _cputime();
   running = true;
   }

void timer::stop()
   {
   assert(running);
   wall = _wallclock() - wall;
   cpu = _cputime() - cpu;
   running = false;
   }

double timer::elapsed() const
   {
   if (running)
      return (_wallclock() - wall);
   return wall;
   }

double timer::cputime() const
   {
   if (running)
      return (_cputime() - cpu);
   return cpu;
   }

double timer::usage() const
   {
   return 100.0 * cputime() / elapsed();
   }

// static functions

std::string timer::format(const double elapsedtime)
   {
   const int max = 256;
   static char tempstring[max];

   if (elapsedtime < 60)
      {
      int order = int(ceil(-log10(elapsedtime) / 3.0));
      if (order > 3)
         order = 3;
      sprintf(tempstring, "%0.2f", elapsedtime * pow(10.0, order * 3));
      switch (order)
         {
         case 0:
            strcat(tempstring, "s");
            break;
         case 1:
            strcat(tempstring, "ms");
            break;
         case 2:
            strcat(tempstring, "us");
            break;
         case 3:
            strcat(tempstring, "ns");
            break;
         }
      }
   else
      {
      int days, hrs, min, sec;

      sec = int(floor(elapsedtime));
      min = sec / 60;
      sec = sec % 60;
      hrs = min / 60;
      min = min % 60;
      days = hrs / 24;
      hrs = hrs % 24;
      if (days > 0)
         sprintf(tempstring, "%d %s, %02d:%02d:%02d", days, (days == 1 ? "day"
               : "days"), hrs, min, sec);
      else
         sprintf(tempstring, "%02d:%02d:%02d", hrs, min, sec);
      }

   return tempstring;
   }

std::string timer::date()
   {
   const int max = 256;
   static char d[max];

   time_t t1 = time(NULL);
   struct tm *t2 = localtime(&t1);
   strftime(d, max, "%d %b %Y, %H:%M:%S", t2);

   return d;
   }

} // end namespace
