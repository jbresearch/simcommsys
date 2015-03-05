/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "image.h"
#include "cputimer.h"
#include "filter/limitfilter.h"

#include <boost/program_options.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <typeinfo>

namespace improcess {

//! load an image from an input stream
template <class S>
libimage::image<S> loadimage(std::istream& sin)
   {
   libimage::image<S> im;
   im.serialize(sin);
   libbase::verifycomplete(sin);
   return im;
   }

/*!
 * \brief   Image sub-sampling process
 * \author  Johann Briffa
 *
 */

template <class S>
void subsample(const int xoff, const int xinc, const int yoff, const int yinc,
      std::istream& sin = std::cin, std::ostream& sout = std::cout)
   {
   // Load input image
   libimage::image<S> image_in = loadimage<S> (sin);
   // Tell use what we're doing
   std::cerr << "Subsampling: " << xoff << ',' << xinc << ',' << yoff << ','
         << yinc << std::endl;
   // Determine size of resulting image
   const int rows = (image_in.size().rows() - yoff + (yinc - 1)) / yinc;
   const int cols = (image_in.size().cols() - xoff + (xinc - 1)) / xinc;
   const int channels = image_in.channels();
   // Create output image
   libimage::image<S> image_out(rows, cols, channels, image_in.range());
   // Repeat for all image channels
   for (int c = 0; c < channels; c++)
      {
      // Extract channel
      libbase::matrix<S> a = image_in.getchannel(c);
      // Do the subsampling process
      libbase::matrix<S> b(rows, cols);
      for (int i = yoff, ii = 0; i < a.size().rows(); i += yinc, ii++)
         for (int j = xoff, jj = 0; j < a.size().cols(); j += xinc, jj++)
            b(ii, jj) = a(i, j);
      // Copy result into output image
      image_out.setchannel(c, b);
      }
   // Save the resulting image
   image_out.serialize(sout);
   }

// Rounding

template <class real>
inline int round(real x)
   {
   return int(floor(x + 0.5));
   }

// Normalized sinc function

template <class real>
inline real sinc(real x)
   {
   const real pi = real(libbase::PI);
   if (x != 0)
      return sin(pi * x) / (pi * x);
   return 1;
   }

template <class real>
inline real sinc(real x, real y)
   {
   return sinc(x) * sinc(y);
   }

// Lanczos impulse response

template <class real>
inline real lanczos(real x, int a)
   {
   if (x > -a && x < a)
      return sinc(x) * sinc(x / a);
   return 0;
   }

template <class real>
inline real lanczos(real x, real y, int a)
   {
   return lanczos(x, a) * lanczos(y, a);
   }

// Resample image using Lanczos filter

template <class S, class real>
real computeat(int i, int j, const libbase::matrix<S>& x, const real xoff,
      const real yoff, const real R, const int a)
   {
   // determine input and output image sizes
   const int xrows = x.size().rows();
   const int xcols = x.size().cols();
   // compute sample
   real yy = 0;
   const int imin = std::max(0, int(ceil(i / R - xoff - a)));
   const int imax = std::min(xrows - 1, int(floor(i / R - xoff + a)));
   const int jmin = std::max(0, int(ceil(j / R - yoff - a)));
   const int jmax = std::min(xcols - 1, int(floor(j / R - yoff + a)));
   for (int ii = imin; ii <= imax; ii++)
      for (int jj = jmin; jj <= jmax; jj++)
         yy += x(ii, jj) * lanczos<real> (i / R - xoff - ii, j / R - yoff - jj,
               a);
   return yy;
   }

template <class S, class real>
libbase::matrix<S> resample(const libbase::matrix<S>& x, const real xoff,
      const real yoff, const real R, const int a)
   {
   std::cerr << "Lanczos resampling (type " << typeid(real).name()
         << ", off = (" << xoff << "," << yoff << "), R = " << R << ", a = "
         << a << ")" << std::endl;
   // determine input and output image sizes
   const int xrows = x.size().rows();
   const int xcols = x.size().cols();
   const int yrows = round(xrows * R);
   const int ycols = round(xcols * R);
   std::cerr << "Channel: " << xcols << "×" << xrows << " -> " << ycols << "×"
         << yrows << std::endl;
   // create destination image
   libbase::matrix<S> y(yrows, ycols);
   // iterate through all destination pixels and compute
   for (int i = 0; i < yrows; i++)
      for (int j = 0; j < ycols; j++)
         // TODO: remove round() for non-int types
         y(i, j) = S(round(computeat<S, real> (i, j, x, xoff, yoff, R, a)));
   // end
   return y;
   }

/*!
 * \brief   Image re-sampling process using Lanczos filter
 * \author  Johann Briffa
 *
 */

template <class S, class real>
void resample(const real xoff, const real yoff, const real scale,
      const int limit, std::istream& sin = std::cin, std::ostream& sout =
            std::cout)
   {
   // Load input image
   libimage::image<S> image_in = loadimage<S> (sin);
   // Tell use what we're doing
   std::cerr << "Resampling: " << xoff << ',' << yoff << ',' << scale << ','
         << limit << std::endl;
   // Create output image
   libimage::image<S> image_out;
   // Process each channel
   for (int c = 0; c < image_in.channels(); c++)
      {
      libbase::matrix<S> plane_in = image_in.getchannel(c);
      libbase::matrix<S> plane_out = resample<S, real> (plane_in, xoff, yoff,
            scale, limit);
      if (c == 0)
         image_out.resize(plane_out.size().rows(), plane_out.size().cols(),
               image_in.channels());
      // Limit values to usable range
      libimage::limitfilter<S> filter(image_in.lo(), image_in.hi());
      filter.apply(plane_out, plane_out);
      // Copy into output image
      image_out.setchannel(c, plane_out);
      }
   // Save the resulting image
   image_out.serialize(sout);
   }

/*!
 * \brief   Safe auto-scale process
 * \author  Johann Briffa
 *
 * Auto-scales the contrast in the image, forcing the scaling factor to be
 * a power of 2. This ensures that the bits in the pixel values keep their
 * statistics.
 *
 * \note This template expects 'S' to be an integer type.
 */

template <class S>
void safescale(std::istream& sin = std::cin, std::ostream& sout = std::cout)
   {
   // Load input image
   libimage::image<S> image_in = loadimage<S> (sin);
   // Tell use what we're doing
   std::cerr << "Safe auto-scale: ";
   // Determine size of resulting image
   const int rows = image_in.size().rows();
   const int cols = image_in.size().cols();
   const int channels = image_in.channels();
   const int range = image_in.range();
   // Create output image
   libimage::image<S> image_out(rows, cols, channels, range);
   // Determine the largest pixel value in the input image
   S maxval = 0;
   for (int c = 0; c < channels; c++)
      maxval = std::max(maxval, image_in.getchannel(c).max());
   const S depth = int(ceil(log2(maxval + 1)));
   std::cerr << "used bit depth = " << depth << ", ";
   // Determine scaling factor and apply
   const S scaling = (1 << int(floor(log2(range / maxval))));
   std::cerr << "scaling factor = " << scaling << std::endl;
   for (int c = 0; c < channels; c++)
      {
      // Extract channel
      libbase::matrix<S> a = image_in.getchannel(c);
      // Apply scaling
      a *= scaling;
      // Copy result into output image
      image_out.setchannel(c, a);
      }
   // Save the resulting image
   image_out.serialize(sout);
   }

/*!
 * \brief   Image Processing Command
 * \author  Johann Briffa
 */

int main(int argc, char *argv[])
   {
   libbase::cputimer tmain("Main timer");

   // Set up user parameters
   namespace po = boost::program_options;
   po::options_description desc("Allowed options");
   desc.add_options()("help,h", "print this help message");
   desc.add_options()("type,t", po::value<std::string>()->default_value("int"),
         "image pixel type");
   desc.add_options()("input,i", po::value<std::string>(),
         "input filename (stdin if absent)");
   desc.add_options()("output,o", po::value<std::string>(),
         "output filename (stdout if absent)");
   desc.add_options()("subsample", po::value<std::string>(),
         "subsampling pattern (xoff,xinc,yoff,yinc)");
   desc.add_options()("resample", po::value<std::string>(),
         "resampling pattern (xoff,yoff,scale,limit)");
   desc.add_options()("safescale", "auto-scales contrast by a power of 2");
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   po::notify(vm);

   // Validate user parameters
   if (vm.count("help") || (vm.count("subsample") + vm.count("resample")
         + vm.count("safescale")) != 1)
      {
      std::cerr << desc << std::endl;
      return 1;
      }
   // Shorthand access for parameters
   const std::string type = vm["type"].as<std::string> ();
   // Interpret input/output filenames
   std::fstream fin, fout;
   if (vm.count("input") > 0)
      fin.open(vm["input"].as<std::string> ().c_str(), std::ios::in
            | std::ios::binary);
   if (vm.count("output") > 0)
      fout.open(vm["output"].as<std::string> ().c_str(), std::ios::out
            | std::ios::binary);
   // Choose between given files and standard I/O
   std::istream& sin = fin.is_open() ? fin : std::cin;
   std::ostream& sout = fout.is_open() ? fout : std::cout;

   if (vm.count("subsample") == 1)
      {
      const std::string parameters = vm["subsample"].as<std::string> ();
      // Process parameters
      std::istringstream ssin(parameters);
      char c;
      int xoff, xinc, yoff, yinc;
      ssin >> xoff >> c >> xinc >> c >> yoff >> c >> yinc;

      // Main process
      if (type == "int")
         subsample<int> (xoff, xinc, yoff, yinc, sin, sout);
      else if (type == "float")
         subsample<float> (xoff, xinc, yoff, yinc, sin, sout);
      else if (type == "double")
         subsample<double> (xoff, xinc, yoff, yinc, sin, sout);
      else
         {
         std::cerr << "Unrecognized pixel type: " << type << std::endl;
         return 1;
         }
      }
   else if (vm.count("resample") == 1)
      {
      const std::string parameters = vm["resample"].as<std::string> ();
      // Process parameters
      std::istringstream ssin(parameters);
      char c;
      double xoff, yoff, scale;
      int limit;
      ssin >> xoff >> c >> yoff >> c >> scale >> c >> limit;

      // Main process
      if (type == "int")
         resample<int, float> (float(xoff), float(yoff), float(scale), limit,
               sin, sout);
      else if (type == "float")
         resample<float, float> (float(xoff), float(yoff), float(scale), limit,
               sin, sout);
      else if (type == "double")
         resample<double, double> (xoff, yoff, scale, limit, sin, sout);
      else
         {
         std::cerr << "Unrecognized pixel type: " << type << std::endl;
         return 1;
         }
      }
   else if (vm.count("safescale") == 1)
      {
      // Main process
      if (type == "int")
         safescale<int> (sin, sout);
      else
         {
         std::cerr << "Unsupported pixel type: " << type << std::endl;
         return 1;
         }
      }

   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return improcess::main(argc, argv);
   }
