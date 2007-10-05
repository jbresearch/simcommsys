#ifndef __masterslave_h
#define __masterslave_h

#include "config.h"
#include "vcs.h"
#include "vector.h"
#include "socket.h"
#include "timer.h"
#include "functor.h"
#include <map>

/*
  Version 1.00 (20-21 Apr 2007)
  * new class to support socket-based master-slave relationship
  * derived from cmpi 2.40
  * supports dynamic slave list
  * meant to replace cmpi in montecarlo
  * TODO (2): serialize to network byte order always...
  * TODO (3): eventually cmpi will be modified to support this class interface model,
    and a new abstract class created to encapsulate both models
  
  Version 1.10 (23-25 Apr 2007)
  * Modified all send/receive functions to use network byte order, allowing
    heterogenous usage
  * Added functions to send/receive byte-wide buffers & strings
  * Changed collection of CPU usage to CPU time
  * Modified waitforevent() by adding a bool parameter, to be able to disable the
    acceptance of new connections
  * Changed disable() from a static function to a regular member, and added automatic
    calling from the object destructor
  * Fixed CPU usage information reporting, by implementing the transfer between slaves
    and master, through a new function called updatecputime()
  * Left passing of priority to enable function as default priority, but this can
    now be overridden by a command-line parameter
  * Changed usage model so that client functions are not statics, and so that users
    of this class now declare themselves as derived classes, rather than instantiating
    and object; this is tied with the requirements for RPC functions.
  * TODO: In view of above, most functions are now protected rather than public, since
    only enable/disable are required by other than the derived classes.
  * Changed function-call model so that we don't have to pass pointers; this was
    in great part necessitated by the above change, since the current model only supports
    global pointers. Instead, function calls are now done by passing a string reference,
    which is used as a key in a map list. Two new functions have been added:
    - fregister() to allow registering of functions by derived classes, and
    - fcall() to actually call them
    Since this class cannot know the exact type of the function pointers, these are held
    by functors, implemented as an abstract base class and a templated derived one.
  * Heavily refactored
  
  Version 1.20 (8 May 2007)
  * Ported to Windows, using Winsock2 API
  * TODO: make setting priority effective on Windows
*/

namespace libbase {

class masterslave {
   static const vcs version;
   // constants (tags)
   static const int tag_getname;
   static const int tag_getcputime;
   static const int tag_work;
   static const int tag_die;

// communication objects
public:
   class slave {
      friend class masterslave;
   protected:
      socket *sock;
      enum { NEW, EVENT_PENDING, IDLE, WORKING } state;
   };

// items for use by everyone (?)
private:
   std::map<std::string, functor *> fmap;
   bool initialized;
   double cputimeused;
   timer t;
protected:
   void fregister(const std::string& name, functor *f); 
   void fcall(const std::string& name);
public:
   // global enable of cluster system
   void enable(int *argc, char **argv[], const int priority=10);
   // informative functions
   bool isenabled() const { return initialized; };
   double getcputime() const { return cputimeused; };
   int getnumslaves() const { return smap.size(); };

// items for use by slaves
private:
   libbase::socket *master;
   // helper functions
   void close(libbase::socket *s);
   void setpriority(const int priority);
   void connect(const std::string& hostname, const int16u port);
   std::string gethostname();
   int gettag();
   void sendname();
   void sendcputime();
   void slaveprocess(const std::string& hostname, const int16u port, const int priority);
public:
   // slave -> master communication
   bool send(const void *buf, const size_t len);
   bool send(const int x);
   bool send(const double x);
   bool send(const vector<double>& x);
   bool send(const std::string& x);
   bool receive(void *buf, const size_t len);
   bool receive(int& x);
   bool receive(double& x);
   bool receive(std::string& x);

// items for use by master
private:
   std::map<socket *, slave *> smap;
   // helper functions
   void close(slave *s);
public:
   // creation and destruction
   masterslave();
   ~masterslave();
   // disable process
   void disable();
   // slave-interface functions
   slave *newslave();
   slave *idleslave();
   slave *pendingslave();
   bool anyoneworking();
   void waitforevent(const bool acceptnew=true);
   void resetslaves();
   // master -> slave communication
   bool send(slave *s, const void *buf, const size_t len);
   bool send(slave *s, const int x);
   bool send(slave *s, const double x);
   bool send(slave *s, const std::string& x);
   bool call(slave *s, const std::string& x);
   bool updatecputime(slave *s);
   bool receive(slave *s, void *buf, const size_t len);
   bool receive(slave *s, int& x);
   bool receive(slave *s, double& x);
   bool receive(slave *s, vector<double>& x);
   bool receive(slave *s, std::string& x);
};

}; // end namespace

#endif
