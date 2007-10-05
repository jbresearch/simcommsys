#include "channel.h"
#include "serializer.h"

namespace libcomm {

const libbase::vcs channel::version("Channel Base module (channel)", 1.40);

// transmission function

void channel::transmit(const libbase::vector<sigspace>& tx, libbase::vector<sigspace>& rx)
   {
   // Initialize results vector
   rx.init(tx);
   // Corrupt the modulation symbols (simulate the channel)
   for(int i=0; i<tx.size(); i++)
      rx(i) = corrupt(tx(i));
   }

// serialization functions

std::ostream& operator<<(std::ostream& sout, const channel* x)
   {
   sout << x->name() << "\n";
   x->serialize(sout);
   return sout;
   }

std::istream& operator>>(std::istream& sin, channel*& x)
   {
   std::string name;
   sin >> name;
   x = (channel*) libbase::serializer::call("channel", name);
   if(x == NULL)
      {
      std::cerr << "FATAL ERROR (channel): Type \"" << name << "\" unknown.\n";
      exit(1);
      }
   x->serialize(sin);
   return sin;
   }

}; // end namespace
