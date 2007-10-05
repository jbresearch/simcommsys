/* timer
 * ~~~~~
 * a class which can be used to time subroutines, etc. resolution is in microseconds,
 * but the class can also handle durations in weeks/years/anything you care to time...
 */

#ifndef __timer_h
#define __timer_h

#include "config.h"
#include "vcs.h"
#include <iostream.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

extern const vcs timer_version;

/*
  Version 1.10 (6 Mar 1999)
  added a new function elapsed(), which returns the number of seconds elapsed.

  Version 1.11 (26 Apr 1999)
  added a new static function date(), which returns the current date and time.

  Version 1.12 (22 Jul 1999)
  added a new function usage(), which returns the percentage CPU usage.
*/
class timer {
   char *name;
   double wall, cpu;
   bool running;
   inline double convert(const struct timeval& tv) const;
   inline double wallclock() const;
   inline double cputime() const;
public:
   inline timer(char *n = NULL);
   inline ~timer();
   inline void start();
   inline void stop();
   inline void divide(int n);
   inline double elapsed() const;
   inline double usage() const;
   friend ostream& operator<<(ostream& s, timer& t);

   static char *date();
};

inline double timer::convert(const struct timeval& tv) const
   {
   return((double)tv.tv_sec + (double)tv.tv_usec * 1E-6);
   }
   
inline double timer::wallclock() const
   {
   struct timeval	tv;
   struct timezone	tz;

   gettimeofday(&tv, &tz);

   return convert(tv);
   }
 
inline double timer::cputime() const
   {
   struct rusage	usage;
   double		cpu;

   getrusage(RUSAGE_SELF, &usage);
   cpu = convert(usage.ru_utime);
   getrusage(RUSAGE_CHILDREN, &usage);
   cpu += convert(usage.ru_utime);

   return(cpu);
   }

inline timer::timer(char *n)
   {
   if(n == NULL)
      name = NULL;
   else
      {
      name = new char[strlen(n)+1];
      strcpy(name, n);
      }
   start();
   }

inline timer::~timer()
   {
   if(running)
      {
      cerr << "Timer";
      if(name != NULL)
         cerr << " (" << name << ")";
      cerr << " expired after " << *this << "\n";
      }
   if(name != NULL)
      delete[] name;
   }
   
inline void timer::start()
   {
   wall = wallclock();
   cpu = cputime();
   running = true;
   }  
   
inline void timer::stop()
   {
   if(!running)
      {
      cerr << "Warning: tried to stop a timer that was not running.\n";
      return;
      }
   wall = wallclock() - wall;
   cpu = cputime() - cpu;
   running = false;
   }
   
inline void timer::divide(int n)
   {
   if(running)
      {
      cerr << "Warning: tried to divide a running timer.\n";
      return;
      }
   wall /= double(n);
   cpu /= double(n);
   }

inline double timer::elapsed() const
   {
   if(running)
      return(wallclock() - wall);
   return wall;
   }

inline double timer::usage() const
   {
   double _wall, _cpu;

   if(running)
      {
      _wall = wallclock() - wall;
      _cpu = cputime() - cpu;
      }
   else
      {
      _wall = wall;
      _cpu = cpu;
      }
   return 100.0*_cpu/_wall;
   }

#endif
