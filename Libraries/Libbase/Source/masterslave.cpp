/*!
   \file

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$
*/

#include "masterslave.h"

#include "timer.h"
#include "pacifier.h"
#include <iostream>
#include <sstream>

#ifdef WIN32
#include <winsock2.h>
#else
#include <unistd.h>
#include <signal.h>
#endif

#ifndef HOST_NAME_MAX
#  define HOST_NAME_MAX 255
#endif

namespace libbase {

using std::cerr;
using std::clog;
using std::flush;

// items for use by everyone

void masterslave::fregister(const std::string& name, functor *f)
   {
   trace << "DEBUG: Register function \"" << name << "\" - ";
   fmap[name] = f;
   trace << fmap.size() << " functions registered, done.\n";
   }

void masterslave::fcall(const std::string& name)
   {
   trace << "DEBUG: Call function \"" << name << "\" - ";
   functor *f = fmap[name];
   if(f == NULL)
      {
      cerr << "Function \"" << name << "\" unknown, cannot continue.\n";
      exit(1);
      }
   f->call();
   trace << "done.\n";
   }

/*! \brief Global enable/disable of master-slave system

   \note Endpoint can be:
         - 'local', indicating local-computation model; in this case, the
           class will not be initialized.
         - ':port', indicating server-mode, bound to given port
         - 'hostname:port', indicating client-mode, connecting to given
           host/port combination
*/
void masterslave::enable(const std::string& endpoint, bool quiet, int priority)
   {
   assert(!initialized);

#ifndef WIN32
   signal(SIGPIPE, SIG_IGN);
#endif

   // hostname is the part before the ':', or the whole string if there is no ':'
   const size_t n = endpoint.find(':');
   const std::string hostname = endpoint.substr(0,n);
   // port is the part after the ':', if there was one
   // if there is no port, hostname must be 'local'
   int port=0;
   if(n != std::string::npos)
      std::istringstream(endpoint.substr(n+1)) >> port;
   // interpret quiet flag
   if(quiet)
      pacifier::disable_output();
   // Handle option for local computation only
   if(hostname.compare("local") == 0 && port == 0)
      return;
   // If the hostname part isn't empty, it's a slave process
   if(hostname.length() > 0)
      slaveprocess(hostname, port, priority);
   // Otherwise, this must be the master process.
   master = new socket;
   assertalways(master->bind(port));
   trace << "Master system bound to port " << port << "\n";

   initialized = true;
   }


// static items (for use by slaves)

void masterslave::close(libbase::socket *s)
   {
   cerr << "Losing connection with master [" << s->getip() << ":" << s->getport() << "]\n";
   delete s;
   s = NULL;
   }

void masterslave::setpriority(const int priority)
   {
#ifdef WIN32
#else
   const int PRIO_CURRENT = 0;
   ::setpriority(PRIO_PROCESS, PRIO_CURRENT, priority);
#endif
   }

void masterslave::connect(const std::string& hostname, const int16u port)
   {
   cerr << "Connecting to " << hostname << ":" << port << "\n";
   master = new socket;
   if(!master->connect(hostname, port))
      {
      cerr << "Connection failed, giving up.\n";
      exit(1);
      }
   }

std::string masterslave::gethostname()
   {
   const int len = HOST_NAME_MAX+1;
   char hostname[len];
   ::gethostname(hostname, len);
   return hostname;
   }

int masterslave::gettag()
   {
   timer t;
   int tag;
   if(!receive(tag))
      {
      cerr << "Connection failed waiting for tag, dying here...\n";
      exit(1);
      }
   t.stop();
   trace << "Slave latency = " << t << ": ";
   return tag;
   }

void masterslave::sendname()
   {
   std::string hostname = gethostname();
   if(!send(hostname))
      {
      cerr << "Connection failed sending hostname, dying here...\n";
      exit(1);
      }
   trace << "send hostname [" << hostname << "]\n" << flush;
   }

void masterslave::sendcputime()
   {
   const double cputime = t.cputime();
   t.start();
   if(!send(cputime))
      {
      cerr << "Connection failed sending CPU time, dying here...\n";
      exit(1);
      }
   cputimeused += cputime;
   trace << "send usage [" << cputime << "]\n" << flush;
   }

void masterslave::dowork()
   {
   std::string key;
   if(!receive(key))
      {
      cerr << "Connection failed waiting for function reference, dying here...\n";
      exit(1);
      }
   trace << "system working\n" << flush;
   fcall(key);
   }

void masterslave::slaveprocess(const std::string& hostname, const int16u port, const int priority)
   {
   setpriority(priority);
   connect(hostname, port);
   // Status information for user
   cerr << "Slave system starting at priority " << priority << ".\n";
   // infinite loop, until we are explicitly told to die
   t.start();
   while(true)
      switch(int tag = gettag())
         {
         case GETNAME:
            sendname();
            break;
         case GETCPUTIME:
            sendcputime();
            break;
         case WORK:
            dowork();
            break;
         case DIE:
            t.stop();
            cerr << "Received die request, stopping after " << timer::format(getcputime()) << " CPU runtime (" << int(100*getusage()) << "%% usage).\n";
            close(master);
            exit(0);
         default:
            cerr << "received bad tag [" << tag << "]\n" << flush;
            exit(1);
         }
   }

// slave -> master communication

bool masterslave::send(const void *buf, const size_t len)
   {
   if(!master->insistwrite(buf, len))
      {
      close(master);
      return false;
      }
   return true;
   }


/*! \brief Send a vector<double> to the master
   \note Vector size is sent first; this makes foreknowledge of size and
         pre-initialization unnecessary.
*/
bool masterslave::send(const vector<double>& x)
   {
   // determine and send vector size first
   const int count = x.size();
   if(!send(count))
      return false;
   // copy vector elements to an array and send at once
   double *a = new double[count];
   for(int i=0; i<count; i++)
      a[i] = x(i);
   const bool success = send(a, sizeof(double)*count);
   delete[] a;
   return success;
   }

bool masterslave::send(const std::string& x)
   {
   int len = int(x.length());
   if(!send(len))
      return false;
   return send(x.c_str(), len);
   }

bool masterslave::receive(void *buf, const size_t len)
   {
   if(!master->insistread(buf, len))
      {
      close(master);
      return false;
      }
   return true;
   }

bool masterslave::receive(std::string& x)
   {
   int len;
   if(!receive(len))
      return false;
   char *buf = new char[len];
   const bool success = receive(buf, len);
   if(success)
      x.assign(buf, len);
   delete[] buf;
   return success;
   }


// non-static items (for use by master)

void masterslave::close(slave *s)
   {
   cerr << "Slave [" << s->sock->getip() << ":" << s->sock->getport() << "] gone";
   smap.erase(s->sock);
   delete s->sock;
   delete s;
   cerr << ", currently have " << smap.size() << " clients\n";
   }

// creation and destruction

masterslave::masterslave()
   {
   initialized = false;
   cputimeused = 0;
   master = NULL;
   }

masterslave::~masterslave()
   {
   disable();
   }

// disable function

/*! \brief Shuts down master-slave system

   \todo Specify what happens if the system was never initialized
*/
void masterslave::disable()
   {
   // if the master-slave system is not initialized, there is nothing to do
   if(!initialized)
      return;

   // kill all remaining slaves
   clog << "Killing idle slaves:" << flush;
   while(slave *s = idleslave())
      {
      trace << "DEBUG (disable): Idle slave found (" << s << "), killing.\n";
      clog << "." << flush;
      send(s, int(DIE));
      }
   clog << " done\n";
   // print timer information
   t.stop();
   clog << "Time elapsed: " << t << "\n" << flush;
   clog << "CPU usage on master: " << int(t.usage()) << "%\n" << flush;
   clog.precision(2);
   clog << "Average speedup factor: " << getusage() << "\n" << flush;
   // update flag
   initialized = false;
   }

// slave-interface functions

masterslave::slave *masterslave::newslave()
   {
   for(std::map<socket *, slave *>::iterator i=smap.begin(); i != smap.end(); ++i)
      if(i->second->state == slave::NEW)
         {
         i->second->state = slave::IDLE;
         return i->second;
         }
   return NULL;
   }

masterslave::slave *masterslave::idleslave()
   {
   for(std::map<socket *, slave *>::iterator i=smap.begin(); i != smap.end(); ++i)
      if(i->second->state == slave::IDLE)
         {
         i->second->state = slave::WORKING;
         return i->second;
         }
   return NULL;
   }

masterslave::slave *masterslave::pendingslave()
   {
   for(std::map<socket *, slave *>::iterator i=smap.begin(); i != smap.end(); ++i)
      if(i->second->state == slave::EVENT_PENDING)
         {
         i->second->state = slave::IDLE;
         return i->second;
         }
   return NULL;
   }

/*! \brief Number of slaves currently in 'working' state
*/
int masterslave::workingslaves() const
   {
   int count = 0;
   for(std::map<socket *, slave *>::const_iterator i=smap.begin(); i != smap.end(); ++i)
      if(i->second->state == slave::WORKING)
         count++;
   return count;
   }

bool masterslave::anyoneworking() const
   {
   for(std::map<socket *, slave *>::const_iterator i=smap.begin(); i != smap.end(); ++i)
      if(i->second->state == slave::WORKING)
         return true;
   return false;
   }

/*! \brief Waits for a socket event
   \param acceptnew Flag to indicate whether new connections are allowed
                  (defaults to true)
   \param timeout Return with no event if this many seconds elapses (zero
                  means wait forever; this is the default)
*/
void masterslave::waitforevent(const bool acceptnew, const double timeout)
   {
   static bool firsttime = true;
   if(firsttime)
      {
      cerr << "Master system ready; waiting for clients.\n";
      firsttime = false;
      }

   static bool signalentry = true;
   if(signalentry)
      {
      trace << "DEBUG (estimate): Waiting for event.\n";
      signalentry = false;
      }

   // create list of sockets and select
   std::list<socket *> sl, al;

   sl.push_back(master);
   for(std::map<socket *, slave *>::iterator i=smap.begin(); i != smap.end(); ++i)
      sl.push_back(i->second->sock);

   al = socket::select(sl, timeout);
   if(!al.empty())
      signalentry = true;
   for(std::list<socket *>::iterator i=al.begin(); i != al.end(); ++i)
      {
      if((*i)->islistener() && acceptnew)
         {
         slave *newslave = new slave;
         newslave->sock = (*i)->accept();
         newslave->state = slave::NEW;
         smap[newslave->sock] = newslave;
         cerr << "New slave [" << newslave->sock->getip() << ":" << newslave->sock->getport() << "], currently have " << smap.size() << " clients\n";
         }
      else
         {
         slave *j = smap[*i];
         j->state = slave::EVENT_PENDING;
         }
      }
   }

/*!
   \brief Reset given slave to the 'new' state

   \note Slave must be in the 'idle' state
*/
void masterslave::resetslave(slave *s)
   {
   assertalways(s->state == slave::IDLE);
   s->state = slave::NEW;
   }

/*!
   \brief Reset all 'idle' slaves to the 'new' state
*/
void masterslave::resetslaves()
   {
   while(slave *s = idleslave())
      s->state = slave::NEW;
   }

// master -> slave communication

bool masterslave::send(slave *s, const void *buf, const size_t len)
   {
   if(!s->sock->insistwrite(buf, len))
      {
      close(s);
      return false;
      }
   return true;
   }

bool masterslave::send(slave *s, const std::string& x)
   {
   int len = int(x.length());
   if(!send(s, len))
      return false;
   return send(s, x.c_str(), len);
   }

/*! \brief Accumulate CPU time for given slave
   \param s Slave from which to get CPU time
*/
bool masterslave::updatecputime(slave *s)
   {
   double cputime;
   if(!send(s, int(GETCPUTIME)) || !receive(s, cputime) )
      return false;
   cputimeused += cputime;
   return true;
   }

bool masterslave::receive(slave *s, void *buf, const size_t len)
   {
   if(!s->sock->insistread(buf, len))
      {
      close(s);
      return false;
      }
   return true;
   }

/*! \brief Receive a vector<double> from given slave
   \note Vector size is obtained first; this makes foreknowledge of size and
         pre-initialization unnecessary.
*/
bool masterslave::receive(slave *s, vector<double>& x)
   {
   // get vector size first
   int count;
   if(!receive(s, count))
      return false;
   // get vector elements
   double *a = new double[count];
   if(!receive(s, a, sizeof(double)*count))
      return false;
   // initialize vector and copy elements over
   x.assign(a, count);
   delete[] a;
   return true;
   }

bool masterslave::receive(slave *s, std::string& x)
   {
   int len;
   if(!receive(s,len))
      return false;
   char *buf = new char[len];
   const bool success = receive(s, buf, len);
   if(success)
      x.assign(buf, len);
   delete[] buf;
   return success;
   }

}; // end namespace
