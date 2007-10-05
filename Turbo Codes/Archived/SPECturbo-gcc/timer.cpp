#include "timer.h"

const vcs timer_version("Timer module (timer)", 1.12);

ostream& operator<<(ostream& s, timer& t)
   {
   int		days, hrs, min;
   double	sec;
   
   if(t.running)
      t.stop();
   
   int flags = s.flags();
   
   if(t.wall < 60)
      {
      s.setf(ios::fixed, ios::floatfield);
      s.precision(2);
      int order = int(ceil(-log10(t.wall)/3.0));
      if(order > 3)
         order = 3;
      s << t.wall * pow(10, order*3);
      switch(order)
         {
         case 0:
            s << "s";
            break;
         case 1:
            s << "ms";
            break;
         case 2:
            s << "us";
            break;
         case 3:
            s << "ns";
            break;
         }
      }
   else
      {
      min = int(floor(t.wall / 60.0));
      sec = t.wall - (min * 60.0);
      hrs = min / 60;
      min = min % 60;
      days = hrs / 24;
      hrs = hrs % 24;
      if(days > 0)
         s << days << (days==1 ? " day, " : " days, ");
      s.width(2);
      s.fill('0');
      s << hrs << ":";
      s.width(2);
      s.fill('0');
      s << min << ":";
      s.setf(ios::fixed, ios::floatfield);
      s.precision(1);
      s.width(4);
      s.fill('0');
      s << sec;
      s << " (CPU " << int(100.0*t.cpu/t.wall) << "%)";      
      }
   
   s.flags(flags);
   
   return s;
   }

char *timer::date()
   {
   const int max = 256;
   static char d[max];

   time_t t1 = time(NULL);
   struct tm *t2 = localtime(&t1);
   strftime(d, max, "%d %b %Y, %H:%M:%S", t2);

   return d;
   }

