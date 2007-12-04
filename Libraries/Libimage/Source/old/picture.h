#ifndef __picture_h
#define __picture_h

#include "config.h"
#include "vcs.h"
#include "matrix.h"
#include "pixel.h"

/*
  Version 1.01 (6 Mar 2002)
  changed vcs version variable from a global to a static class variable.
  also changed use of iostream from global to std namespace.

  Version 1.10 (10 Nov 2006)
  * defined class and associated data within "libimage" namespace.
  * removed use of "using namespace std", replacing by tighter "using" statements as needed.
*/

namespace libimage {

class picture {
   static const libbase::vcs version;
   libbase::matrix<pixel> data;
public:
   picture();
   ~picture();

   picture(const libbase::matrix<double>& in);

   bool load(const char *fname);
   bool save(const char *fname);

   void quantise(const int bpc);

   pixel operator()(int x, int y) const { return data(x,y); };
   pixel& operator()(int x, int y) { return data(x,y); };

   libbase::matrix<double> luminance() const;

   int width() const { return data.xsize(); };
   int height() const { return data.ysize(); };
};

}; // end namespace

#endif
