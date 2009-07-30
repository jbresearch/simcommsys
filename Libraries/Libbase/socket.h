#ifndef __socket_h
#define __socket_h

#include "config.h"

#include <string>
#include <list>

namespace libbase {

/*!
 * \brief   Networking sockets.
 * \author  Johann Briffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 * 
 * \version 1.00 (27 Jun 2002)
 * original version; just a placeholder
 * 
 * \version 1.10 (18-19 Apr 2007)
 * - defined class and associated data within "libbase" namespace.
 * - removed use of "using namespace std", replacing by tighter "using" statements as needed.
 * - Integrated code from worker pool project by:
 * - Johann Briffa   <j.briffa@ieee.org>
 * - Vangelis Koukis <vkoukis@cslab.ece.ntua.gr>
 * - Currently supports POSIX; eventually need to port to Windows
 * 
 * \version 1.11 (21 Apr 2007)
 * - added getter properties for ip & hostname
 * 
 * \version 1.12 (23 Apr 2007)
 * - split read/write functions so that the normal functions return the size
 * as usual, while the insist functions merely return true/false; this simplifies
 * use in masterslave class
 * - modified write to only require a const void * pointer to the buffer, since
 * this function should never modify the contents
 * - split io() function due to inconsistent requirements for buffer pointer
 * 
 * \version 1.20 (8 May 2007)
 * - Ported class to Windows, using Winsock2 API
 * 
 * \version 1.21 (20 Nov 2007)
 * - Added timeout facility to select(), defaulting to no-timeout
 * 
 * \version 1.22 (28 Nov 2007)
 * - modifications to silence 64-bit portability warnings
 * - explicit conversion from size_t to int in io()
 * - ditto in bind(), accept() and connect()
 */

class socket {
   // constant values - client
   static const int connect_tries;
   static const int connect_delay;
#ifdef WIN32
   // static values - object count
   static int objectcount;
#endif
   // internal variables
   int sd;
   std::string ip;
   int16u port;
   bool listener;
private:
   // helper functions
   template <class T> ssize_t io(T buf, size_t len);
   template <class T> ssize_t insistio(T buf, size_t len);
public:
   // constructor/destructor
   socket();
   ~socket();
   // listener property
   bool islistener() const
      {
      return listener;
      }
   // wait for client connects
   bool bind(int16u port);
   static std::list<socket *> select(std::list<socket *> sl,
         const double timeout = 0);
   socket *accept();
   // open connection to server
   bool connect(std::string hostname, int16u port);
   // read/write data
   ssize_t write(const void *buf, size_t len);
   ssize_t read(void *buf, size_t len);
   bool insistwrite(const void *buf, size_t len);
   bool insistread(void *buf, size_t len);
   // get ip & hostname
   std::string getip() const
      {
      return ip;
      }
   int16u getport() const
      {
      return port;
      }
};

} // end namespace

#endif
