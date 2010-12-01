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
 * 
 * \section svn Version Control
 * - $Id$
 */

#include "image.h"
#include "timer.h"

#include <boost/program_options.hpp>
#include <iostream>
#include <sstream>
#include <typeinfo>

namespace improcess {

//! load an image from an input stream
template <class S>
libimage::image<S> loadimage(std::istream& sin)
   {
   libimage::image<S> im;
   im.serialize(sin);
   libbase::verifycompleteload(sin);
   return im;
   }

/*!
 * \brief   Main process
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 */

template <class S>
void process(const int xoff, const int xinc, const int yoff, const int yinc,
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
      // Copy result into stego-image
      image_out.setchannel(c, b);
      }
   // Save the resulting image
   image_out.serialize(sout);
   }

/*!
 * \brief   Stego-System Embedder
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 */

int main(int argc, char *argv[])
   {
   libbase::timer tmain("Main timer");

   // Set up user parameters
   namespace po = boost::program_options;
   po::options_description desc("Allowed options");
   desc.add_options()("help,h", "print this help message");
   desc.add_options()("type,t", po::value<std::string>()->default_value("int"),
         "image pixel type");
   desc.add_options()("subsample", po::value<std::string>(),
         "subsampling pattern (xoff,xinc,yoff,yinc)");
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   po::notify(vm);

   // Validate user parameters
   if (vm.count("help") || vm.count("subsample") == 0)
      {
      std::cerr << desc << std::endl;
      return 1;
      }
   // Shorthand access for parameters
   const std::string type = vm["type"].as<std::string> ();
   const std::string subsample = vm["subsample"].as<std::string> ();
   // Process parameters
   std::istringstream sin(subsample);
   char c;
   int xoff, xinc, yoff, yinc;
   sin >> xoff >> c >> xinc >> c >> yoff >> c >> yinc;

   // Main process
   if (type == "int")
      process<int> (xoff, xinc, yoff, yinc);
   else if (type == "float")
      process<float> (xoff, xinc, yoff, yinc);
   else if (type == "double")
      process<double> (xoff, xinc, yoff, yinc);
   else
      {
      std::cerr << "Unrecognized pixel type: " << type << std::endl;
      return 1;
      }

   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return improcess::main(argc, argv);
   }
