#ifndef __awgn_h
#define __awgn_h

#include "config.h"
#include "vcs.h"
#include "channel.h"
#include "randgen.h"
#include "itfunc.h"
#include <math.h>

extern const vcs awgn_version;

/* Version 1.10 (15 Apr 1999)
   Changed the definition of set_snr to avoid using the pow() function.
   This was causing an unexplained SEGV with optimised code
*/
class awgn : public virtual channel {
   randgen		r;
   double		sigma, Eb, No, snr_db;
public:
   awgn();
   void seed(const int32u s);
   void set_eb(const double Eb);
   void set_snr(const double snr_db);
   double get_snr() const;
   sigspace corrupt(const sigspace& s);
   double pdf(const sigspace& tx, const sigspace& rx) const;
};

inline void awgn::seed(const int32u s)
   {
   r.seed(s);
   }

inline void awgn::set_eb(const double Eb)
   {
   // Eb is the signal energy for each bit duration
   awgn::Eb = Eb;
   sigma = sqrt(Eb*No);
   }

inline void awgn::set_snr(const double snr_db)
   {
   awgn::snr_db = snr_db;
   // No is half the noise energy/modulation symbol for a normalised signal
   No = 0.5*exp(-snr_db/10.0 * log(10.0));
   sigma = sqrt(Eb*No);
   }
   
inline double awgn::get_snr() const
   {
   return snr_db;
   }

inline awgn::awgn()
   {
   awgn::Eb = 1;
   awgn::set_snr(0);
   awgn::seed(0);
   }
   
inline sigspace awgn::corrupt(const sigspace& s)
   {
   const double x = r.gval(sigma);
   const double y = r.gval(sigma);
   return s + sigspace(x, y);
   }

inline double awgn::pdf(const sigspace& tx, const sigspace& rx) const
   {      
   sigspace n = rx - tx;
   return gauss(n.i() / sigma) * gauss(n.q() / sigma);
   }

#endif

