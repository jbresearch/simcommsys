#include "image.h"
#include "itfunc.h"

namespace libimage {

const libbase::vcs image::version("Digital Image module (image)", 1.00);

using libbase::round;
using libbase::trace;

// Freeimage sub-class

bool image::freeimage::initialized = false;
image::freeimage image::library;

// constructor / destructor

image::freeimage::freeimage()
   {
   assert(!initialized);
        // call this ONLY when linking with FreeImage as a static library
#ifdef FREEIMAGE_LIB
   //trace << "DEBUG (image): Initialising FreeImage library.\n";
        FreeImage_Initialise();
#endif // FREEIMAGE_LIB
   initialized = true;
   // fill in I/O structures
   in.read_proc  = (FI_ReadProc) &read;
   in.write_proc = (FI_WriteProc) NULL;
   in.seek_proc  = (FI_SeekProc) &iseek;
   in.tell_proc  = (FI_TellProc) &itell;
   out.read_proc  = (FI_ReadProc) NULL;
   out.write_proc = (FI_WriteProc) &write;
   out.seek_proc  = (FI_SeekProc) &oseek;
   out.tell_proc  = (FI_TellProc) &otell;
   }

image::freeimage::~freeimage()
   {
        // call this ONLY when linking with FreeImage as a static library
#ifdef FREEIMAGE_LIB
   //trace << "DEBUG (image): DeInitialising FreeImage library.\n";
        FreeImage_DeInitialise();
#endif // FREEIMAGE_LIB
   }

// translations between handle I/O and stream I/O

unsigned DLL_CALLCONV image::freeimage::read(void *buffer, unsigned size, unsigned count, fi_handle handle)
   {
   //trace << "DEBUG (image): read " << size << "x" << count << " into address " << std::hex << int(buffer) << std::dec << "\n";
   std::istream *sin = (std::istream *) handle;
   sin->read((char *)buffer, size*count);
   return sin->gcount()/size;
   }

int DLL_CALLCONV image::freeimage::iseek(fi_handle handle, long offset, int origin)
   {
   //trace << "DEBUG (image): iseek " << offset << " from ";
   std::istream *sin = (std::istream *) handle;
   std::ios_base::seekdir way;
   switch(origin)
      {
      case SEEK_SET:
         //trace << "start\n";
         way = std::ios_base::beg;
         break;
      case SEEK_CUR:
         //trace << "current\n";
         way = std::ios_base::cur;
         break;
      case SEEK_END:
         //trace << "end\n";
         way = std::ios_base::end;
         break;
      default:
         assert("ERROR (image): origin unknown.");
      }
   sin->seekg(offset, way);
   return 0;
   }

long DLL_CALLCONV image::freeimage::itell(fi_handle handle)
   {
   //trace << "DEBUG (image): itell - ";
   std::istream *sin = (std::istream *) handle;
   std::streampos p = sin->tellg();
   //trace << p << "\n";
   return p;
   }

unsigned DLL_CALLCONV image::freeimage::write(void *buffer, unsigned size, unsigned count, fi_handle handle)
   {
   //trace << "DEBUG (image): write " << size << "x" << count << " from address " << std::hex << int(buffer) << std::dec << "\n";
   std::ostream *sout = (std::ostream *) handle;
   sout->write((const char *)buffer, size*count);
   return count;
   }

int DLL_CALLCONV image::freeimage::oseek(fi_handle handle, long offset, int origin)
   {
   //trace << "DEBUG (image): oseek " << offset << " from ";
   std::ostream *sout = (std::ostream *) handle;
   std::ios_base::seekdir way;
   switch(origin)
      {
      case SEEK_SET:
         //trace << "start\n";
         way = std::ios_base::beg;
         break;
      case SEEK_CUR:
         //trace << "current\n";
         way = std::ios_base::cur;
         break;
      case SEEK_END:
         //trace << "end\n";
         way = std::ios_base::end;
         break;
      default:
         assert("ERROR (image): origin unknown.");
      }
   sout->seekp(offset, way);
   return 0;
   }

long DLL_CALLCONV image::freeimage::otell(fi_handle handle)
   {
   //trace << "DEBUG (image): otell - ";
   std::ostream *sout = (std::ostream *) handle;
   std::streampos p = sout->tellp();
   //trace << p << "\n";
   return p;
   }

// other freeimage interface functions

void image::unload()
   {
   if(dib != NULL)
      {
      FreeImage_Unload(dib);
      dib = NULL;
      }
   }

double image::getpixel(BYTE *pixel, FREE_IMAGE_TYPE type, int bpp, unsigned channel) const
   {
   switch(type)
      {
      case FIT_BITMAP:
         switch(bpp)
            {
            case 8:
               assert(channel == 0);
               return pixel[0] / maxval(8);
            case 24:
               {
               RGBTRIPLE *d = (RGBTRIPLE *)pixel;
               switch(channel)
                  {
                  case 0:
                     return d->rgbtRed / maxval(8);
                  case 1:
                     return d->rgbtGreen / maxval(8);
                  case 2:
                     return d->rgbtBlue / maxval(8);
                  default:
                     assert("ERROR (image): invalid channel number.");
                  }
               }
            case 32:
               {
               RGBQUAD *d = (RGBQUAD *)pixel;
               switch(channel)
                  {
                  case 0:
                     return d->rgbRed / maxval(8);
                  case 1:
                     return d->rgbGreen / maxval(8);
                  case 2:
                     return d->rgbBlue / maxval(8);
                  case 3:
                     return d->rgbReserved / maxval(8);
                  default:
                     assert("ERROR (image): invalid channel number.");
                  }
               }
            default:
               assert("ERROR (image): invalid bpp value.");
            }
      case FIT_RGB16:
         {
         FIRGB16 *d = (FIRGB16 *)pixel;
         switch(channel)
            {
            case 0:
               return d->red / maxval(5);
            case 1:
               return d->green / maxval(5);
            case 2:
               return d->blue / maxval(5);
            default:
               assert("ERROR (image): invalid channel number.");
            }
         }
      case FIT_RGBA16:
         {
         FIRGBA16 *d = (FIRGBA16 *)pixel;
         switch(channel)
            {
            case 0:
               return d->red / maxval(5);
            case 1:
               return d->green / maxval(5);
            case 2:
               return d->blue / maxval(5);
            case 3:
               return d->alpha / maxval(5);
            default:
               assert("ERROR (image): invalid channel number.");
            }
         }
      default:
         assert("ERROR (image): image type unknown.");
      }
   return -1;
   }

void image::setpixel(BYTE *pixel, FREE_IMAGE_TYPE type, int bpp, unsigned channel, double value)
   {
   switch(type)
      {
      case FIT_BITMAP:
         switch(bpp)
            {
            case 8:
               assert(channel == 0);
               pixel[0] = BYTE(round(value * maxval(8)));
               break;
            case 24:
               {
               RGBTRIPLE *d = (RGBTRIPLE *)pixel;
               switch(channel)
                  {
                  case 0:
                     d->rgbtRed = BYTE(round(value * maxval(8)));
                     break;
                  case 1:
                     d->rgbtGreen = BYTE(round(value * maxval(8)));
                     break;
                  case 2:
                     d->rgbtBlue = BYTE(round(value * maxval(8)));
                     break;
                  default:
                     assert("ERROR (image): invalid channel number.");
                  }
               }
            case 32:
               {
               RGBQUAD *d = (RGBQUAD *)pixel;
               switch(channel)
                  {
                  case 0:
                     d->rgbRed = BYTE(round(value * maxval(8)));
                     break;
                  case 1:
                     d->rgbGreen = BYTE(round(value * maxval(8)));
                     break;
                  case 2:
                     d->rgbBlue = BYTE(round(value * maxval(8)));
                     break;
                  case 3:
                     d->rgbReserved = BYTE(round(value * maxval(8)));
                     break;
                  default:
                     assert("ERROR (image): invalid channel number.");
                  }
               }
            default:
               assert("ERROR (image): invalid bpp value.");
            }
      case FIT_RGB16:
         {
         FIRGB16 *d = (FIRGB16 *)pixel;
         switch(channel)
            {
            case 0:
               d->red = WORD(round(value * maxval(5)));
               break;
            case 1:
               d->green = WORD(round(value * maxval(5)));
               break;
            case 2:
               d->blue = WORD(round(value * maxval(5)));
               break;
            default:
               assert("ERROR (image): invalid channel number.");
            }
         }
      case FIT_RGBA16:
         {
         FIRGBA16 *d = (FIRGBA16 *)pixel;
         switch(channel)
            {
            case 0:
               d->red = WORD(round(value * maxval(5)));
               break;
            case 1:
               d->green = WORD(round(value * maxval(5)));
               break;
            case 2:
               d->blue = WORD(round(value * maxval(5)));
               break;
            case 3:
               d->alpha = WORD(round(value * maxval(5)));
               break;
            default:
               assert("ERROR (image): invalid channel number.");
            }
         }
      default:
         assert("ERROR (image): image type unknown.");
      }
   }

// Construction / destruction

image::image()
   {
   dib = NULL;
   }

image::~image()
   {
   unload();
   }

// Saving/loading functions

std::ostream& image::serialize(std::ostream& sout) const
   {
   // set format/compression/quality according to what user wants
   FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
   int flags = 0;
   switch(format)
      {
      case tiff:
         trace << "DEBUG (image): Writing image in TIFF format.\n";
         fif = FIF_TIFF;
         switch(compression)
            {
            case none:
               trace << "DEBUG (image): TIFF format - no compression.\n";
               flags = TIFF_NONE;
               break;
            case lzw:
               trace << "DEBUG (image): TIFF format - LZW compression.\n";
               flags = TIFF_LZW;
               break;
            default:
               trace << "DEBUG (image): TIFF format - unknown compression setting.\n";
            }
         break;
      case jpeg:
         trace << "DEBUG (image): Writing image in JPEG format (Q = " << quality << ").\n";
         fif = FIF_JPEG;
         flags = quality;
         break;
      default:
         trace << "DEBUG (image): unkown image format.\n";
      }
   // save the iamge
   FreeImage_SaveToHandle(fif, dib, &library.out, (fi_handle) &sout, flags);
   return sout;
   }

std::istream& image::serialize(std::istream& sin)
   {
   FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
   fif = FreeImage_GetFileTypeFromHandle(&library.in, (fi_handle) &sin);
   // confirm that we can identify the format and that we know how to read it
   assert(fif != FIF_UNKNOWN);
   assert(FreeImage_FIFSupportsReading(fif));
   // set format/compression/quality according to what was loaded
   switch(fif)
      {
      case FIF_TIFF:
         trace << "DEBUG (image): Reading image in TIFF format.\n";
         format = tiff;
         break;
      case FIF_JPEG:
         trace << "DEBUG (image): Reading image in JPEG format.\n";
         format = jpeg;
         break;
      case FIF_PNG:
         trace << "DEBUG (image): Reading image in PNG format.\n";
         format = png;
         break;
      default:
         trace << "DEBUG (image): Format type (" << fif << ") not handled.\n";
      }
   // unload any previous image
   unload();
   // load the image
   dib = FreeImage_LoadFromHandle(fif, &library.in, (fi_handle) &sin);
   assert(dib != NULL);
   trace << "DEBUG (image): Read image size (" << width() << "x" << height() << "x" << channels() << ").\n";
   return sin;
   }

std::ostream& operator<<(std::ostream& sout, const image& x)
   {
   x.serialize(sout);
   return sout;
   }

std::istream& operator>>(std::istream& sin, image& x)
   {
   x.serialize(sin);
   return sin;
   }

// Informative functions

int image::width() const
   {
   return FreeImage_GetWidth(dib);
   };

int image::height() const
   {
   return FreeImage_GetHeight(dib);
   };

int image::channels() const
   {
   switch(FreeImage_GetColorType(dib))
      {
      case FIC_MINISBLACK:
      case FIC_MINISWHITE:
         return 1;
      case FIC_RGB:
         return 3;
      case FIC_RGBALPHA:
      case FIC_CMYK:
         return 4;
      }
   return 0;
   };

// Conversion to/from matrix of pixels

libbase::matrix<double> image::getchannel(int c) const
   {
   assert(c >= 0);
   assert(c < channels());
   // get information on image format
   unsigned pitch = FreeImage_GetPitch(dib);
   FREE_IMAGE_TYPE type = FreeImage_GetImageType(dib);
   int bpp = FreeImage_GetBPP(dib);
   BYTE *bits = (BYTE *)FreeImage_GetBits(dib);
   // create matrix
   libbase::matrix<double> m;
   m.init(width(), height());
   // loop through all pixels
   for(int y=0; y<height(); y++, bits+=pitch)
      {
      BYTE *pixel = bits;
      for(int x=0; x<width(); x++, pixel+=(bpp>>3))
         m(x,y) = getpixel(pixel, type, bpp, c);
      }
   return m;
   }

void image::setchannel(int c, const libbase::matrix<double>& m)
   {
   assert(c >= 0);
   assert(c < channels());
   assert(m.size().rows() == width());
   assert(m.size().cols() == height());
   // get information on image format
   unsigned pitch = FreeImage_GetPitch(dib);
   FREE_IMAGE_TYPE type = FreeImage_GetImageType(dib);
   int bpp = FreeImage_GetBPP(dib);
   BYTE *bits = (BYTE *)FreeImage_GetBits(dib);
   // loop through all pixels
   for(int y=0; y<height(); y++, bits+=pitch)
      {
      BYTE *pixel = bits;
      for(int x=0; x<width(); x++, pixel+=(bpp>>3))
         setpixel(pixel, type, bpp, c, m(x,y));
      }
   }

//image::operator libbase::matrix<double>() const
//image& image::operator=(const libbase::matrix<double>& m)

}; // end namespace
