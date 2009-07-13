#ifndef __awfilter_h
#define __awfilter_h

#include "filter.h"
#include "rvstatistics.h"

/*
 Version 1.00 (4 Apr 2002)
 Initial version. Implements Lee's algorithm (Lee, 1980), as used in Matlab's wiener2
 filter function (in the image processing library). Note that Matlab provides a way for
 the function to estimate the noise variance itself - this is actually computed as the
 mean value of the image local variance. This class allows this to be done by setting
 the noise parameter to zero. The estimator function is also publicly available.

 Version 1.10 (17 Oct 2002)
 class is now derived from filter. Also modified the global threshold estimator to
 utilize the multi-pass architecture defined in filter.

 Version 1.11 (1 Nov 2002)
 added display update during update and process functions, after every line is completed.

 Version 1.20 (3 Mar 2003)
 added function that returns noise threshold estimate to allow feedback.

 Version 1.21 (13 Oct 2006)
 added explicit type conversion in process, when assigning value to output matrix.

 Version 1.30 (10 Nov 2006)
 * defined class and associated data within "libimage" namespace.
 */

namespace libimage {

extern const libbase::vcs awfilter_version;

template <class T> class awfilter : public filter<T> {
protected:
   // user-supplied settings
   int m_d;
   double m_noise;
   // internal variables
   libbase::rvstatistics rvglobal;
public:
   awfilter()
      {
      }
   awfilter(const int d, const double noise)
      {
      init(d, noise);
      }
   virtual ~awfilter()
      {
      }
   // initialization
   void init(const int d, const double noise);
   // progress display
   void display_progress(const int done, const int total) const
      {
      }
   // parameter estimation (updates internal statistics)
   void reset();
   void update(const libbase::matrix<T>& in);
   void estimate();
   double get_estimate() const
      {
      return m_noise;
      }
   // filter process loop (only updates output matrix)
   void process(const libbase::matrix<T>& in, libbase::matrix<T>& out) const;
};

} // end namespace

#endif
