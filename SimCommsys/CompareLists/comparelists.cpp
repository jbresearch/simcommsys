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

#include "levenshtein.h"
#include "hamming.h"
#include "cputimer.h"

#include <boost/program_options.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <algorithm>

namespace comparelists {

using std::cout;
using std::cerr;
namespace po = boost::program_options;

// coordinate object

template <class T>
class coordinate {
   T x; //!< x-coordinate
   T y; //!< y-coordinate
   static T t2; // square-distance threshold for equality
public:
   coordinate() :
      x(0), y(0)
      {
      }
   explicit coordinate(const T x, const T y) :
      x(x), y(y)
      {
      }
   bool operator==(const coordinate& other) const
      {
      const T d2 = square(other.x - x) + square(other.y - y);
      return d2 <= t2;
      }
   friend T diff2(const coordinate& a, const coordinate& b)
      {
      const T d2 = square(a.x - b.x) + square(a.y - b.y);
      return d2;
      }
   static void set_threshold(const T d)
      {
      assert(d >= 0);
      t2 = square(d);
      }
   friend std::ostream& operator<<(std::ostream& s, const coordinate<T>& p)
      {
      s << p.x << "\t" << p.y;
      return s;
      }
   friend std::istream& operator>>(std::istream& s, coordinate<T>& p)
      {
      s >> p.x >> p.y;
      return s;
      }
};

// definition of static member variables
template <>
double coordinate<double>::t2 = 0;

// single-dimensional object

template <class T>
class value {
   T x; //!< value
   static T t2; // square-distance threshold for equality
public:
   value() :
      x(0)
      {
      }
   explicit value(const T x) :
      x(x)
      {
      }
   bool operator==(const value& other) const
      {
      const T d2 = square(other.x - x);
      return d2 <= t2;
      }
   friend T diff2(const value& a, const value& b)
      {
      const T d2 = square(a.x - b.x);
      return d2;
      }
   static void set_threshold(const T d)
      {
      assert(d >= 0);
      t2 = square(d);
      }
   friend std::ostream& operator<<(std::ostream& s, const value<T>& p)
      {
      s << p.x;
      return s;
      }
   friend std::istream& operator>>(std::istream& s, value<T>& p)
      {
      s >> p.x;
      return s;
      }
};

// definition of static member variables
template <>
double value<double>::t2 = 0;

// sequence loading

template <class T>
std::vector<T> getsequence(const std::string& fname)
   {
   cerr << "Loading sequence from \"" << fname << "\"...";
   // load sequence from file
   std::vector<T> sequence;
   std::ifstream file(fname.c_str(), std::ios_base::in | std::ios_base::binary);
   // load elements until we hit end of file, skipping comments and whitespace
   T element;
   while (file >> libbase::eatcomments >> element)
      sequence.push_back(element);
   cerr << "done, length = " << sequence.size() << std::endl;
   return sequence;
   }

// compute number of common elements

template <class T>
int repeatability(const std::vector<T>& s1, const std::vector<T>& s2)
   {
   // copy vectors to lists
   std::list<T> t1, t2;
   t1.resize(s1.size());
   t2.resize(s2.size());
   std::copy(s1.begin(), s1.end(), t1.begin());
   std::copy(s2.begin(), s2.end(), t2.begin());
   // do the comparison
   int count = 0;
   for (typename std::list<T>::iterator i1 = t1.begin(); i1 != t1.end(); i1++)
      {
      typename std::list<T>::iterator i2 = find(t2.begin(), t2.end(), *i1);
      if (i2 != t2.end())
         {
         count++;
         // remove matching value
         t2.erase(i2);
         // confirm this was unique
         if (find(t2.begin(), t2.end(), *i1) != t2.end())
            {
            std::cerr << "Ambiguity found for element (" << *i1 << ")"
                  << std::endl;
            exit(1);
            }
         }
      }
   return count;
   }

// compute mean-square error for common elements

template <class T>
double compute_mse(const std::vector<T>& s1, const std::vector<T>& s2)
   {
   // copy vectors to lists
   std::list<T> t1, t2;
   t1.resize(s1.size());
   t2.resize(s2.size());
   std::copy(s1.begin(), s1.end(), t1.begin());
   std::copy(s2.begin(), s2.end(), t2.begin());
   // do the comparison
   int count = 0;
   double mse = 0;
   for (typename std::list<T>::iterator i1 = t1.begin(); i1 != t1.end(); i1++)
      {
      typename std::list<T>::iterator i2 = find(t2.begin(), t2.end(), *i1);
      if (i2 != t2.end())
         {
         count++;
         mse += diff2(*i1, *i2);
         // remove matching value
         t2.erase(i2);
         // confirm this was unique
         if (find(t2.begin(), t2.end(), *i1) != t2.end())
            {
            std::cerr << "Ambiguity found for element (" << *i1 << ")"
                  << std::endl;
            exit(1);
            }
         }
      }
   return mse / double(count);
   }

// main process

template <class T>
void process(const std::string& fname1, const std::string& fname2,
      const double t, const bool ham, const bool lev, const bool rep,
      const bool mse)
   {
   // Load sequences and user settings
   const std::vector<T> s1 = getsequence<T> (fname1);
   const std::vector<T> s2 = getsequence<T> (fname2);
   T::set_threshold(t);

   // Do the computations
   if (ham)
      {
      const int hd = libbase::hamming<T>(libbase::vector<T>(s1),
            libbase::vector<T>(s2));
      cout << "Hamming distance = " << hd << std::endl;
      }
   if (lev)
      {
      const int ld = libbase::levenshtein<T>(libbase::vector<T>(s1),
            libbase::vector<T>(s2));
      cout << "Levenshtein distance = " << ld << std::endl;
      }
   if (rep)
      {
      const int count = repeatability<T> (s1, s2);
      cout << "Repeatability = " << count << std::endl;
      }
   if (mse)
      {
      const double mse = compute_mse<T> (s1, s2);
      cout << "Mean-square error = " << mse << std::endl;
      }
   }

/*!
 * \brief   Compare two sequences
 * \author  Johann Briffa
 *
 * Comparison can be one of:
 * - Levenshtein distance (# of ins/del/sub)
 * - Repeatability (# of common elements, unordered)
 * - Mean Square Error (for equivalent elements)
 */

int main(int argc, char *argv[])
   {
   libbase::cputimer tmain("Main timer");

   // Set up user parameters
   po::options_description desc("Allowed options");
   desc.add_options()("help", "print this help message");
   desc.add_options()("sequence1-file,i", po::value<std::string>(),
         "input file containing first sequence");
   desc.add_options()("sequence2-file,j", po::value<std::string>(),
         "input file containing second sequence");
   desc.add_options()("threshold,t", po::value<double>()->default_value(0.5),
         "threshold for Euclidian distance");
   desc.add_options()("coordinates,c", po::bool_switch(),
         "read elements as coordinate pairs");
   desc.add_options()("hamming,h", po::bool_switch(),
         "compute Hamming distance");
   desc.add_options()("levenshtein,l", po::bool_switch(),
         "compute Levenshtein distance");
   desc.add_options()("repeatability,r", po::bool_switch(),
         "compute number of common elements");
   desc.add_options()("mse,m", po::bool_switch(),
         "compute mean-square error between common elements");
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   po::notify(vm);

   // read switch parameters
   const bool ham = vm["hamming"].as<bool> ();
   const bool lev = vm["levenshtein"].as<bool> ();
   const bool rep = vm["repeatability"].as<bool> ();
   const bool mse = vm["mse"].as<bool> ();

   // Validate user parameters
   if (vm.count("help") || vm.count("sequence1-file") == 0 || vm.count(
         "sequence1-file") == 0 || (!ham && !lev && !rep && !mse))
      {
      cout << desc << std::endl;
      return 0;
      }

   // read parameters
   const std::string fname1 = vm["sequence1-file"].as<std::string> ();
   const std::string fname2 = vm["sequence2-file"].as<std::string> ();
   const double t = vm["threshold"].as<double> ();
   const bool coordinates = vm["coordinates"].as<bool> ();

   // call main process

   if (coordinates)
      process<coordinate<double> > (fname1, fname2, t, ham, lev, rep, mse);
   else
      process<value<double> > (fname1, fname2, t, ham, lev, rep, mse);

   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return comparelists::main(argc, argv);
   }
