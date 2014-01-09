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

#include "config.h"

/* Define the version of Windows required (assume that this will work with
 * the last version)
 * NOTE: Moving the definition to config.h is breaking other files (since this
 *    is included in quite a number of places if not from everywhere). While
 *    this is not the cleanest solution, it is not doing any harm. Please
 *    investigate further before moving.
 */
#ifdef _WIN32
#  ifndef _WIN32_WINNT
#    define _WIN32_WINNT _WIN32_WINNT_MAXVER
#  endif
#endif

#ifdef _WIN32
#  include <afx.h>
#  include <conio.h>
#else
#  include <unistd.h>
#  include <termios.h>
#  include <sys/ioctl.h>
#  include <cerrno>
#  include <cstring>
#endif
#include <csignal>
#include <cstdio>
#include <sstream>

namespace libbase {

// Debugging tools

class tracestreambuf : public std::streambuf {
protected:
   std::string buffer;
public:
   tracestreambuf()
      {
      buffer = "";
      }
   virtual ~tracestreambuf()
      {
      }
   int underflow()
      {
      return EOF;
      }
   int overflow(int c = EOF);
};

inline int tracestreambuf::overflow(int c)
   {
#ifndef NDEBUG
   if (c == '\r' || c == '\n')
      {
      if (!buffer.empty())
         {
#ifdef _WIN32
         TRACE("%s\n", buffer.c_str());
#else
         std::clog << buffer.c_str() << std::endl;
#endif
         buffer = "";
         }
      }
#ifdef _WIN32
   // handle TRACE limit in Windows (512 chars including NULL)

   else if(buffer.length() == 511)
      {
      TRACE("%s", buffer.c_str());
      buffer = c;
      }
#endif
   else
      buffer += c;
#endif
   return 1;
   }

tracestreambuf g_tracebuf;
std::ostream trace(&g_tracebuf);

// Constants

const double PI = 3.14159265358979323846;

#ifdef _WIN32
const char DIR_SEPARATOR = '\\';
#else
const char DIR_SEPARATOR = '/';
#endif

const int ALIGNMENT = 128;

/*! \brief Checks if a key has been pressed
 * \return true if this has happened
 */
int keypressed(void)
   {
#ifdef _WIN32
   return _kbhit();
#else
   int count = 0;
   int error;
   struct timespec tv;
   struct termios otty, ntty;

   tcgetattr(STDIN_FILENO, &otty);
   ntty = otty;
   ntty.c_lflag &= ~ICANON; /* raw mode */

   if (0 == (error = tcsetattr(STDIN_FILENO, TCSANOW, &ntty)))
      {
      error += ioctl(STDIN_FILENO, FIONREAD, &count);
      error += tcsetattr(STDIN_FILENO, TCSANOW, &otty);
      // minimal delay gives up cpu time slice, allows use in a tight loop
      tv.tv_sec = 0;
      tv.tv_nsec = 10;
      nanosleep(&tv, NULL);
      }

   return error == 0 ? count : -1;
#endif
   }

/*! \brief Waits for the user to hit a key and returns its value.
 * \note The user's response is not shown on screen.
 */
int readkey(void)
   {
#ifdef _WIN32
   return _getch();
#else
   unsigned char ch;
   int error;
   struct termios otty, ntty;

   fflush( stdout);
   tcgetattr(STDIN_FILENO, &otty);
   ntty = otty;

   ntty.c_lflag &= ~ICANON; /* line settings   */

   /* disable echoing the char as it is typed */
   ntty.c_lflag &= ~ECHO; /* disable echo        */

   ntty.c_cc[VMIN] = 1; /* block for input  */
   ntty.c_cc[VTIME] = 0; /* timer is ignored */

   // flush the input buffer before blocking for new input
   //#define FLAG TCSAFLUSH
   // return a char from the current input buffer, or block if no input is waiting.
#define FLAG TCSANOW

   if (0 == (error = tcsetattr(STDIN_FILENO, FLAG, &ntty)))
      {
      /* get a single character from stdin */
      error = read(STDIN_FILENO, &ch, 1);
      /* restore old settings */
      error += tcsetattr(STDIN_FILENO, FLAG, &otty);
      }

   return (error == 1 ? (int) ch : -1);
#endif
   }

static bool interrupt_caught = false;

/*! \brief Interrupt-signal handling function
 * This function is meant to catch Ctrl-C during execution, to be used in the
 * same way as keypressed(), allowing pre-mature interruption of running MPI
 * processes (which can't handle keypressed() events).
 *
 * \note The signal handler is set the first time that interrupted() is
 * called; this means that the mechanism is not activated until the
 * first time it is called, which generally works fine as this function
 * is meant to be used within a loop as part of the condition statement.
 */
static void catch_signal(int sig_num)
   {
   trace << "DEBUG (catch_signal): caught signal " << sig_num << std::endl;
   // re-set the signal handler again for next time
   signal(sig_num, catch_signal);
   // update variables accordingly
   if (sig_num == SIGINT)
      {
      std::cerr << "Caught interrupt..." << std::endl;
      interrupt_caught = true;
      }
   }

bool interrupted(void)
   {
   // set the signal handler
   signal(SIGINT, catch_signal);
   // return pre-set variable
   return interrupt_caught;
   }

// System error message reporting

std::string getlasterror()
   {
   std::ostringstream sout;
#ifdef _WIN32
   TCHAR buf[80];
   DWORD code = GetLastError();
   FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM, NULL, code, NULL, buf, 80, NULL);
   sout << buf << " (" << std::hex << code << std::dec << ")";
#else
   sout << strerror(errno) << " (" << std::hex << errno << std::dec << ")";
#endif
   return sout.str();
   }

//! Function to skip over whitespace

std::istream& eatwhite(std::istream& is)
   {
   char c;
   while (is.get(c))
      {
      if (!isspace(c))
         {
         is.putback(c);
         break;
         }
      }
   return is;
   }

//! Function to skip over any combination of comments and whitespace

std::istream& eatcomments(std::istream& is)
   {
   char c;
   while (is.get(c))
      {
      if (c == '#')
         {
         std::string s;
         getline(is, s);
         }
      else if (!isspace(c))
         {
         is.putback(c);
         break;
         }
      }
   return is;
   }

/*!
 * \brief Check for a failure during the last stream input.
 *
 * If the last stream input did not succeed, throws an exception with
 * details on the stream position where this occurred.
 */
void check_failedload(std::istream &is)
   {
   if (is.fail())
      {
      std::ios::iostate state = is.rdstate();
      is.clear();
      std::ostringstream sout;
      sout << "Failure loading object at position " << is.tellg()
            << ", next line:" << std::endl;
      std::string s;
      getline(is, s);
      sout << s;
      is.clear(state);
      throw load_error(sout.str());
      }
   }

/*!
 * \brief Check for a unloaded data on the stream.
 *
 * If there is still data left on the stream (excluding any initial comments),
 * throws an exception with details on the stream position where this occurred.
 * All data left from this position onwards is also returned.
 */
void check_incompleteload(std::istream &is)
   {
   libbase::eatcomments(is);
   if (!is.eof())
      {
      std::ostringstream sout;
      sout << "Incomplete loading, stopped at position " << is.tellg()
            << ", next line:" << std::endl;
      std::string s;
      getline(is, s);
      sout << s;
      throw load_error(sout.str());
      }
   }

/*!
 * \brief Verify that the last stream data item was read without error.
 *
 * If the last stream input failed an error message is shown, and the program
 * is stopped.
 */
std::istream& verify(std::istream& is)
   {
   check_failedload(is);
   return is;
   }

/*!
 * \brief Verify that all stream data was read without error.
 *
 * If the last stream input failed, or if there is still data left on the
 * stream, an error message is shown, and the program is stopped.
 */
std::istream& verifycomplete(std::istream& is)
   {
   check_failedload(is);
   check_incompleteload(is);
   return is;
   }

} // end namespace
