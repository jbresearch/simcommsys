#ifndef __image_h
#define __image_h

#include "config.h"
#include "vcs.h"
#include "matrix.h"

// comment this out if not using the static library
#define FREEIMAGE_LIB
#include "freeimage.h"
#include <iostream>

/*******************************************************************************

 Version 1.00 (7-14 Dec 2006)
 * initial version
 * class meant to encapsulate the data and functions dealing with a single
 image, potentially containing a number of channels.
 * according to common convention in image processing, the origin is at the
 top left, so that row-major order gives the normal raster conversion.

 *******************************************************************************/

namespace libimage {

class image {
   static const libbase::vcs version;
private:
   // FreeImage interface;
   class freeimage {
      static bool initialized;
   public:
      // pre-built I/O structures
      FreeImageIO in, out;
   public:
      // constructor / destructor
      freeimage();
      ~freeimage();
      // translations between handle I/O and stream I/O
static   unsigned DLL_CALLCONV read(void *buffer, unsigned size, unsigned count, fi_handle handle);
   static int DLL_CALLCONV iseek(fi_handle handle, long offset, int origin);
   static long DLL_CALLCONV itell(fi_handle handle);
   static unsigned DLL_CALLCONV write(void *buffer, unsigned size, unsigned count, fi_handle handle);
   static int DLL_CALLCONV oseek(fi_handle handle, long offset, int origin);
   static long DLL_CALLCONV otell(fi_handle handle);
   };
static freeimage library;
// the image object itself
FIBITMAP *dib;
// other freeimage interface functions
double maxval(int bits) const
   {return double((1<<bits) - 1);};
void unload();
double getpixel(BYTE *pixel, FREE_IMAGE_TYPE type, int bpp, unsigned channel) const;
void setpixel(BYTE *pixel, FREE_IMAGE_TYPE type, int bpp, unsigned channel, double value);
public: // File format interface
typedef enum
   {
   tiff = 0,
   jpeg,
   png
   }format_type;
typedef enum
   {
   none = 0,
   lzw
   }compression_type;
private: // File format interface
format_type format;
compression_type compression;
int quality;
public:
// Construction / destruction
image();
~image();

// File format functions
format_type get_format() const
   {return format;};
compression_type get_compression() const
   {return compression;};
int get_quality() const
   {return quality;};
void set_format(format_type format)
   {image::format = format;};
void set_compression(compression_type compression)
   {image::compression = compression;};
void set_quality(int quality)
   {image::quality = quality;};

// Saving/loading functions
std::ostream& serialize(std::ostream& sout) const;
std::istream& serialize(std::istream& sin);

// Informative functions
int width() const;
int height() const;
int channels() const;

// Conversion to/from matrix of pixels
libbase::matrix<double> getchannel(int c) const;
void setchannel(int c, const libbase::matrix<double>& m);
//operator libbase::matrix<double>() const;
//image& operator=(const libbase::matrix<double>& m);

//// Data access
//T operator()(int x, int y) const { return data(x,y); };
//T& operator()(int x, int y) { return data(x,y); };
};

// Non-member, non-friend functions
std::ostream& operator<<(std::ostream& sout, const image& x);
std::istream& operator>>(std::istream& sin, image& x);

} // end namespace

#endif
