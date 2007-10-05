#include "puncture_stipple.h"

const vcs puncture_file_version("Stippled Puncturing System module (puncture_stipple)", 1.00);


puncture_stipple::puncture_stipple(const int tau, const int s)
   {
   puncture::tau = tau;
   puncture::s = s;
   // initialise the pattern matrix
   pattern.init(s, tau);
   for(int i=0; i<s; i++)
      for(int t=0; t<tau; t++)
         pattern(i, t) = (i==0 || (i-1)%(s-1) == t%(s-1));
   // work out the number of symbols that will be transmitted
   // and fill in the position matrix
   count = 0;
   pos.init(s, tau);
   for(int t=0; t<tau; t++)
      for(int i=0; i<s; i++)
         if(pattern(i, t))
            pos(i, t) = count++;
   }

puncture_stipple::~puncture_stipple()
   {
   }

void puncture_stipple::print(ostream& s) const
   {
   s << "Stipple Puncturing";
   }
