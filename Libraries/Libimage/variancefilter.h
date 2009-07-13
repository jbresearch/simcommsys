#ifndef __variancefilter_h
#define __variancefilter_h

#include "filter.h"

/*
 Version 1.00 (30 Nov 2001)
 Initial version - works local variance of matrix, given radius.

 Version 1.10 (17 Oct 2002)
 class is now derived from filter.

 Version 1.20 (10 Nov 2006)
 * defined class and associated data within "libimage" namespace.
 */

namespace libimage {

extern const libbase::vcs variancefilter_version;

template <class T> class variancefilter : public filter<T> {
protected:
   int m_d;
public:
   variancefilter()
      {
      }
   variancefilter(const int d)
      {
      init(d);
      }
   virtual ~variancefilter()
      {
      }
   // initialization
   void init(const int d);
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
};

} // end namespace

#endif
