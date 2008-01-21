/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "config.h"

#ifdef WIN32
#  include <afx.h>
#  include <conio.h>
#else
#  include <unistd.h>
#  include <termios.h>
#  include <sys/ioctl.h>
#  include <errno.h>
#endif
#include <signal.h>
#include <stdlib.h>
#include <string>
#include <sstream>

#include "timer.h"

namespace libbase {

// Debugging tools

void fail(const char *expression, const char *file, int line)
   {
   std::cerr << "ERROR (" << file << " line " << line << "): assertion " << expression << " failed.\n";
   exit(1);
   }

class tracestreambuf : public std::streambuf {
protected:
   std::string buffer;
public:
   tracestreambuf() { buffer = ""; };
   virtual ~tracestreambuf() {};
   int underflow() { return EOF; };
   int overflow(int c=EOF);
};

inline int tracestreambuf::overflow(int c)
   {
#ifndef NDEBUG
   if(c=='\r' || c=='\n')
      {
      if(!buffer.empty())
         {
#ifdef WIN32
         TRACE("%s\n", buffer.c_str());
#else
         std::clog << buffer.c_str() << std::endl << std::flush;
#endif
         buffer = "";
         }
      }
#ifdef WIN32
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
   };

tracestreambuf g_tracebuf;
std::ostream trace(&g_tracebuf);

// Constants

const double PI = 3.14159265358979323846;

#ifdef WIN32
const char DIR_SEPARATOR = '\\';
#else
const char DIR_SEPARATOR = '/';
#endif

// Checks if a key has been pressed and returns true if this has happened.
int keypressed(void)
   {
#ifdef WIN32
   return _kbhit();
#else
   int            count = 0;
   int            error;
   struct timespec tv;
   struct termios  otty, ntty;

   tcgetattr(STDIN_FILENO, &otty);
   ntty = otty;
   ntty.c_lflag          &= ~ICANON; /* raw mode */

   if (0 == (error = tcsetattr(STDIN_FILENO, TCSANOW, &ntty)))
      {
      error        += ioctl(STDIN_FILENO, FIONREAD, &count);
      error        += tcsetattr(STDIN_FILENO, TCSANOW, &otty);
      // minimal delay gives up cpu time slice, allows use in a tight loop
      tv.tv_sec     = 0;
      tv.tv_nsec    = 10;
      nanosleep(&tv, NULL);
      }

   return error == 0 ? count : -1;
#endif
   }

// Waits for the user to hit a key and returns its value.
// The user's response is not shown on screen.
int readkey(void)
   {
#ifdef WIN32
   return _getch();
#else
   unsigned char                ch;
   int          error;
   struct termios       otty, ntty;

   fflush(stdout);
   tcgetattr(STDIN_FILENO, &otty);
   ntty = otty;

   ntty.c_lflag &= ~ICANON;   /* line settings   */

   /* disable echoing the char as it is typed */
   ntty.c_lflag &= ~ECHO;         /* disable echo        */

   ntty.c_cc[VMIN]  = 1;          /* block for input  */
   ntty.c_cc[VTIME] = 0;          /* timer is ignored */

   // flush the input buffer before blocking for new input
   //#define FLAG TCSAFLUSH
   // return a char from the current input buffer, or block if no input is waiting.
   #define FLAG TCSANOW

   if (0 == (error = tcsetattr(STDIN_FILENO, FLAG, &ntty)))
      {
      /* get a single character from stdin */
      error  = read(STDIN_FILENO, &ch, 1 );
      /* restore old settings */
      error += tcsetattr(STDIN_FILENO, FLAG, &otty);
      }

   return (error == 1 ? (int) ch : -1 );
#endif
   }

// Interrupt-signal handling function

static bool interrupt_caught = false;

static void catch_signal(int sig_num)
   {
   trace << "DEBUG (catch_signal): caught signal " << sig_num << "\n.";
   // re-set the signal handler again for next time
   signal(sig_num, catch_signal);
   // update variables accordingly
   if(sig_num == SIGINT)
      {
      std::cerr << "Caught interrupt...\n";
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

// Pacifier output

std::string pacifier(const std::string& description, int complete, int total)
   {
   static timer t("pacifier");
   static int last = 0;
   static size_t characters = 0;
   const int value = int(100*complete/double(total));

   // if we detect that we've started from zero again,
   // reset the timer and don't print anything
   if(complete == 0 || value < last)
      {
      t.start();
      last = value;
      characters = 0;
      return "";
      }
   // if we're at the last step, stop the timer
   // and return enough spaces to overwrite the last output
   if(complete == total)
      {
      t.stop();
      std::string s;
      if(characters > 0)
         {
         s.assign(characters,' ');
         s += '\r';
         }
      return s;
      }
   // otherwise we know we're someway in between...
   // estimate how long this whole stage will take to complete
   const double estimate = t.elapsed()/double(complete)*total;
   // return a blank if there is no change or if this won't take long enough
   if(value == last || estimate < 60)
      return "";
   // create the required string
   std::ostringstream sout;
   // if this is the first time we're printing something, start on a new line
   if(characters == 0)
      sout << '\n';
   sout << description << ": completed " << value << "%, elapsed " << t
        << " of estimated " << timer::format(estimate) << '\r';
   // update tracker
   last = value;
   characters = sout.str().length();
   return sout.str();
   }

std::string pacifier(int complete, int total)
   {
   return pacifier("Process", complete, total);
   }

// System error message reporting

std::string getlasterror()
   {
   std::ostringstream sout;
#ifdef WIN32
   TCHAR buf[80];
   DWORD code = GetLastError();
   FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM, NULL, code, NULL, buf, 80, NULL);
   sout << buf << " (" << std::hex << code << std::dec << ")";
#else
   sout << strerror(errno) << " (" << std::hex << errno << std::dec << ")";
#endif
   return sout.str();
   }

}; // end namespace
