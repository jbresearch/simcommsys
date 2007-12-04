#ifndef __atmfilter_h
#define __atmfilter_h

#include "filter.h"

/*
  Version 1.00 (11 Nov 2001)
  Initial version - changed from a template function into a template class which keeps
  information across calls to the process routine.

  Version 1.01 (4 Apr 2002)
  made the destructor inline; now using namespace std in the implementation file.

  Version 1.10 (17 Oct 2002)
  class is now derived from filter.

  Version 1.11 (1 Nov 2002)
  added display update during process function, after every line is completed.

  Version 1.20 (10 Nov 2006)
  * defined class and associated data within "libimage" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.
*/

namespace libimage {

extern const libbase::vcs atmfilter_version;

template<class T> class atmfilter : public filter<T> {
protected:
   int   m_d;
   int   m_alpha;
public:
   atmfilter() {};
   atmfilter(const int d, const int alpha) { init(d, alpha); };
   virtual ~atmfilter() {};
   // initialization
   void init(const int d, const int alpha);
   // progress display
   void display_progress(const int done, const int total) const {};
   // parameter estimation (updates internal statistics)
   void reset() {};
   void update(const libbase::matrix<T>& in) {};
   void estimate() {};
   // filter process loop (only updates output matrix)
   void process(const libbase::matrix<T>& in, libbase::matrix<T>& out) const;
};

}; // end namespace

#endif
