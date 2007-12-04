#include "picture.h"
#include <iostream>

#ifdef TIFF
#include <tiff.h>
#include <tiffio.h>
#endif

namespace libimage {

using libbase::int8u;

const libbase::vcs picture::version("True Colour Picture module (picture)", 1.10);

picture::picture()
   {
   }

picture::~picture()
   {
   }

picture::picture(const libbase::matrix<double>& in)
   {
   int width = in.xsize();
   int height = in.ysize();
   
   data.init(width, height);
   for(int y=0; y<height; y++)
      for(int x=0; x<width; x++)
         data(x,y) = pixel(in(x,y));
   }

bool picture::load(const char *fname)
   {
#ifdef TIFF
   TIFF *tif = TIFFOpen(fname, "r");
   if(tif == NULL)
      {
      cerr << "ERROR (picture): failed to open TIFF file for reading (" << fname <<")\n";
      return false;
      }
   int width, height;
   TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
   TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
   uint32 *raster = (uint32 *) _TIFFmalloc(width*height*sizeof(uint32));
   if(raster == NULL)
      {
      cerr << "ERROR (picture): failed to allocate memory (" << width*height*sizeof(uint32)/1024 << "K)\n";
      TIFFClose(tif);
      return false;
      }
   if(!TIFFReadRGBAImage(tif, width, height, raster))
      {
      cerr << "ERROR (picture): failed to read TIFF file\n";
      _TIFFfree(raster);
      TIFFClose(tif);
      return false;
      }
   data.init(width, height);
   for(int x=0; x<width; x++)
      for(int y=0; y<height; y++)
         {
         uint32 abgr = raster[(height-1-y)*width+x];
         int8u r = TIFFGetR(abgr);
         int8u g = TIFFGetG(abgr);
         int8u b = TIFFGetB(abgr);
         data(x,y) = pixel(r,g,b);
         }
   _TIFFfree(raster);
   TIFFClose(tif);
   return true;
#else
   return false;
#endif
   }

bool picture::save(const char *fname)
   {
#ifdef TIFF
   TIFF *tif = TIFFOpen(fname, "w");
   if(tif == NULL)
      {
      cerr << "Error (picture): failed to open TIFF file for writing (" << fname <<")\n";
      return false;
      }
   int width = picture::width();
   int height = picture::height();
   TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
   TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
   TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
   TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 3);
   TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
   TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
   TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
   TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
   TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tif, 0));
   int linebytes = 3*width;
   int8u *buf = (int8u *)_TIFFmalloc(min((int)linebytes, (int)TIFFScanlineSize(tif)));
   for(int y=0; y<height; y++)
      {
      for(int x=0; x<width; x++)
         {
         buf[3*x+0] = data(x,y).red();
         buf[3*x+1] = data(x,y).green();
         buf[3*x+2] = data(x,y).blue();
         }
      if(TIFFWriteScanline(tif, buf, y, 0) < 0)
         {
         cerr << "Error (picture): failed to write scanline (" << y << ")\n";
         _TIFFfree(buf);
         TIFFClose(tif);
         return false;
         }
      }
   _TIFFfree(buf);
   TIFFClose(tif);
   return true;
#else
   return false;
#endif
   }

void picture::quantise(const int bpc)
   {
   int width = picture::width();
   int height = picture::height();
   for(int y=0; y<height; y++)
      for(int x=0; x<width; x++)
         data(x,y).quantise(bpc);
   }

libbase::matrix<double> picture::luminance() const
   {
   int width = picture::width();
   int height = picture::height();
   libbase::matrix<double> r(width,height);
   
   for(int y=0; y<height; y++)
      for(int x=0; x<width; x++)
         r(x,y) = data(x,y).luminance();

   return r;
   }

}; // end namespace
