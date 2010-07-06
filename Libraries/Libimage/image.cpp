#include "image.h"

#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>

namespace libimage {

// Saving/loading functions

template <class T>
std::ostream& image<T>::serialize(std::ostream& sout) const
   {
   libbase::trace << "Saving image" << std::flush;
   // header data
   const int chan = channels();
   assert(chan > 0);
   const int rows = m_data(0).size().rows();
   const int cols = m_data(0).size().cols();
   libbase::trace << " (" << cols << "x" << rows << "x" << chan << ")..."
         << std::flush;
   // write file descriptor
   if (chan == 1 && m_maxval == 1)
      sout << "P4" << std::endl; // bitmap
   else if (chan == 1 && m_maxval > 1)
      sout << "P5" << std::endl; // graymap
   else if (chan == 3)
      sout << "P6" << std::endl; // pixmap
   else
      failwith("Image format not supported");
   // write comment
   sout << "# file written by libimage" << std::endl;
   // write image size
   sout << cols << " " << rows << std::endl;
   // if needed, write maxval
   if (chan > 1 || m_maxval > 1)
      sout << m_maxval << std::endl;
   // write image data
   for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++)
         for (int c = 0; c < chan; c++)
            {
            int p;
            if (typeid(T) == typeid(double) || typeid(T) == typeid(float))
               p = int(round(m_data(c)(i, j) * m_maxval));
            else
               p = int(m_data(c)(i, j));
            assert(p >= 0 && p <= m_maxval);
            sout.put(p);
            }
   // done
   libbase::trace << "done" << std::endl;
   return sout;
   }

template <class T>
std::istream& image<T>::serialize(std::istream& sin)
   {
   libbase::trace << "Loading image" << std::flush;
   // header data
   int cols, rows, chan;
   bool binary;
   // read file header
   std::string line;
   std::getline(sin, line);
   // read file descriptor
   int descriptor;
   assert(line[0] == 'P');
   std::istringstream(line.substr(1)) >> descriptor;
   assertalways(descriptor >= 1 && descriptor <= 6);
   // determine the number of channels
   if (descriptor == 3 || descriptor == 6)
      chan = 3;
   else
      chan = 1;
   // determine the data format
   if (descriptor >= 4 || descriptor <= 6)
      binary = true;
   else
      binary = false;
   // skip comments or empty lines
   do
      {
      std::getline(sin, line);
      } while (line.size() == 0 || line[0] == '#');
   // read image size
   std::istringstream(line) >> cols >> rows;
   // if necessary read pixel value range
   if (descriptor == 1 || descriptor == 4)
      {
      m_maxval = 1;
      assert(!binary); // cannot handle binary bitmaps (packed bits)
      }
   else
      {
      std::getline(sin, line);
      std::istringstream(line) >> m_maxval;
      assert(!binary || m_maxval <= 255); // cannot handle 16-bit binary files
      }
   libbase::trace << " (" << cols << "x" << rows << "x" << chan << ")...";
   // set up space to hold image
   m_data.init(chan);
   for (int c = 0; c < chan; c++)
      m_data(c).init(rows, cols);
   // read image data
   for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++)
         for (int c = 0; c < chan; c++)
            {
            if (binary)
               m_data(c)(i, j) = T(sin.get());
            else
               sin >> m_data(c)(i, j);
            assert(m_data(c)(i, j) >= 0 && m_data(c)(i, j) <= m_maxval);
            }
   assertalways(sin);
   // scale down if we're using floating-point
   if (typeid(T) == typeid(double) || typeid(T) == typeid(float))
      for (int c = 0; c < chan; c++)
         m_data(c) /= T(m_maxval);
   // done
   libbase::trace << "done" << std::endl;
   return sin;
   }

// Explicit Realizations

using libbase::serializer;

template class image<int> ;
template <>
const serializer image<int>::shelper("image", "image<int>", image<int>::create);

template class image<float> ;
template <>
const serializer image<float>::shelper("image", "image<float>",
      image<float>::create);

template class image<double> ;
template <>
const serializer image<double>::shelper("image", "image<double>",
      image<double>::create);

} // end namespace
