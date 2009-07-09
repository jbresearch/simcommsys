#ifndef __limiter_h
#define __limiter_h

#include "filter.h"

/*
 Version 1.00 (11 Nov 2001)
 Initial version.

 Version 1.10 (17 Oct 2002)
 class is now derived from filter.

 Version 1.20 (10 Nov 2006)
 * defined class and associated data within "libimage" namespace.
 */

namespace libimage {

extern const libbase::vcs limiter_version;

template <class T> class limiter : public filter<T> {
protected:
   T m_lo;
   T m_hi;
public:
   limiter()
      {
      }
   limiter(const T lo, const T hi)
      {
      init(lo, hi);
      }
   virtual ~limiter()
      {
      }
   // initialization
   void init(const T lo, const T hi);
   // progress display
   void display_progress(const int done, const int total) const
      {
      }
   // parameter estimation (updates internal statistics)
   void reset()
      {
      }
   void update(const libbase::matrix<T>& in)
      {
      }
   void estimate()
      {
      }
   // filter process loop (only updates output matrix)
   void process(const libbase::matrix<T>& in, libbase::matrix<T>& out) const;
   void process(libbase::matrix<T>& m) const;
};

} // end namespace

#endif
