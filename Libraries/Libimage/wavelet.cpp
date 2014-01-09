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

#include "wavelet.h"
#include "itfunc.h"

namespace libimage {

using std::cerr;
using libbase::trace;

using libbase::weight;
using libbase::vector;
using libbase::matrix;

// static helper functions - quadrature filter constructor

vector<double> wavelet::quadrature(const vector<double>& g)
   {
   const int n = g.size();
   vector<double> h(n);
   for (int i = 0; i < n; i++)
      h(n - i - 1) = (i % 2 == 0) ? -g(i) : g(i);
   return h;
   }

// static helper functions - partial transform/inverse

/*
 the wavelet transform matrix is modified to include the interleaving
 stage directly. Thus it becomes (for Daub4):

 c0  c1  c2  c3
 c0  c1  c2  c3
 \\\\\\\
                            c0  c1  c2  c3
 c2  c3                            c0  c1
 -> Half Point
 c3 -c2  c1 -c0
 c3 -c2  c1 -c0
 \\\\\\\
                            c3 -c2  c1 -c0
 c1 -c0                            c3 -c2

 */

void wavelet::partial_transform(const vector<double>& in, vector<double>& out,
      const int n) const
   {
   assert(in.size() == out.size());
   assert(n <= in.size());
   // trap calls where n is too small
   if (n < 4)
      return;
   // initialise workspace (since out & in may be the same vector)
   vector<double> b(n);
   b = 0;
   // set up some constants that we need
   const int nh = n >> 1;
   const int mask = n - 1;
   const int ncof = g.size();
   // do the transform
   for (int i = 0; i < nh; i++)
      for (int j = 0; j < ncof; j++)
         {
         const int k = ((i << 1) + j) & mask;
         b(i) += g(j) * in(k);
         b(i + nh) += h(j) * in(k);
         }
   // copy the result back from workspace
   out.copyfrom(b);
   }

void wavelet::partial_inverse(const vector<double>& in, vector<double>& out,
      const int n) const
   {
   assert(in.size() == out.size());
   assert(n <= in.size());
   // trap calls where n is too small
   if (n < 4)
      return;
   // initialise workspace (since out & in may be the same vector)
   vector<double> b(n);
   b = 0;
   // set up some constants that we need
   const int nh = n >> 1;
   const int mask = n - 1;
   const int ncof = g.size();
   // do the transform
   for (int i = 0; i < nh; i++)
      for (int j = 0; j < ncof; j++)
         {
         const int k = ((i << 1) + j) & mask;
         b(k) += g(j) * in(i) + h(j) * in(i + nh);
         }
   // copy the result back from workspace
   out.copyfrom(b);
   }
/*
 void wavelet::partial_titransform(vector<double>& a, vector<double>& hsr, vector<double>& hsl, vector<double>& lsr, vector<double>& lsl) const
 {
 const int n = a.size();
 // trap calls where n is too small
 if(n < 4)
 return;
 // initialise workspace
 hsr.init(n);
 hsl.init(n);
 lsr.init(n);
 lsl.init(n);
 hsr = hsl = lsr = lsl = 0;
 // set up some constants that we need
 const int nh = n >> 1;
 const int mask = n-1;
 const int ncof = g.size();
 // do the transform
 for(int i=0; i<nh; i++)
 for(int j=0; j<ncof; j++)
 {
 const int kr = ((i<<1) + j) & mask;
 const int kl = ((i<<1) + (n-1) + j) & mask;
 lsr(i) += g(j) * a(kr);
 hsr(i) += h(j) * a(kr);
 lsl(i) += g(j) * a(kl);
 hsl(i) += h(j) * a(kl);
 }
 }
 */
// initialization

void wavelet::init(const int type, const int par)
   {
   switch (type)
      {
      // Haar
      case 0:
         {
         g.init(2);
         g = 1;
         }
         break;
         // Beylkin
      case 1:
         {
         const double f[] = {.099305765374, .424215360813, .699825214057,
               .449718251149, -.110927598348, -.264497231446, .026900308804,
               .155538731877, -.017520746267, -.088543630623, .019679866044,
               .042916387274, -.017460408696, -.014365807969, .010040411845,
               .001484234782, -.002736031626, .000640485329};
         g.assign(f, sizeof(f) / sizeof(f[0]));
         }
         break;
         // Coiflet
      case 2:
         {
         switch (par)
            {
            case 1:
               {
               const double f[] = {.038580777748, -.126969125396,
                     -.077161555496, .607491641386, .745687558934,
                     .226584265197};
               g.assign(f, sizeof(f) / sizeof(f[0]));
               }
               break;
            case 2:
               {
               const double f[] = {.016387336463, -.041464936782,
                     -.067372554722, .386110066823, .812723635450,
                     .417005184424, -.076488599078, -.059434418646,
                     .023680171947, .005611434819, -.001823208871,
                     -.000720549445};
               g.assign(f, sizeof(f) / sizeof(f[0]));
               }
               break;
            case 3:
               {
               const double f[] = {-.003793512864, .007782596426,
                     .023452696142, -.065771911281, -.061123390003,
                     .405176902410, .793777222626, .428483476378,
                     -.071799821619, -.082301927106, .034555027573,
                     .015880544864, -.009007976137, -.002574517688,
                     .001117518771, .000466216960, -.000070983303,
                     -.000034599773};
               g.assign(f, sizeof(f) / sizeof(f[0]));
               }
               break;
            case 4:
               {
               const double f[] = {.000892313668, -.001629492013,
                     -.007346166328, .016068943964, .026682300156,
                     -.081266699680, -.056077313316, .415308407030,
                     .782238930920, .434386056491, -.066627474263,
                     -.096220442034, .039334427123, .025082261845,
                     -.015211731527, -.005658286686, .003751436157,
                     .001266561929, -.000589020757, -.000259974552,
                     .000062339034, .000031229876, -.000003259680,
                     -.000001784985};
               g.assign(f, sizeof(f) / sizeof(f[0]));
               }
               break;
            case 5:
               {
               const double f[] = {-.000212080863, .000358589677,
                     .002178236305, -.004159358782, -.010131117538,
                     .023408156762, .028168029062, -.091920010549,
                     -.052043163216, .421566206729, .774289603740,
                     .437991626228, -.062035963906, -.105574208706,
                     .041289208741, .032683574283, -.019761779012,
                     -.009164231153, .006764185419, .002433373209,
                     -.001662863769, -.000638131296, .000302259520,
                     .000140541149, -.000041340484, -.000021315014,
                     .000003734597, .000002063806, -.000000167408,
                     -.000000095158};
               g.assign(f, sizeof(f) / sizeof(f[0]));
               }
               break;
               // Undefined
            default:
               cerr << "Undefined parameter (" << par << ") for wavelet type ("
                     << type << ")." << std::endl;
               return;
            }
         }
         break;
         // Daubechies
      case 3:
         {
         switch (par)
            {
            case 4:
               {
               const double f[] = {.482962913145, .836516303738, .224143868042,
                     -.129409522551};
               g.assign(f, sizeof(f) / sizeof(f[0]));
               }
               break;
            case 6:
               {
               const double f[] = {.332670552950, .806891509311, .459877502118,
                     -.135011020010, -.085441273882, .035226291882};
               g.assign(f, sizeof(f) / sizeof(f[0]));
               }
               break;
            case 8:
               {
               const double f[] = {.230377813309, .714846570553, .630880767930,
                     -.027983769417, -.187034811719, .030841381836,
                     .032883011667, -.010597401785};
               g.assign(f, sizeof(f) / sizeof(f[0]));
               }
               break;
            case 10:
               {
               const double f[] = {.160102397974, .603829269797, .724308528438,
                     .138428145901, -.242294887066, -.032244869585,
                     .077571493840, -.006241490213, -.012580751999,
                     .003335725285};
               g.assign(f, sizeof(f) / sizeof(f[0]));
               }
               break;
            case 12:
               {
               const double f[] = {.111540743350, .494623890398, .751133908021,
                     .315250351709, -.226264693965, -.129766867567,
                     .097501605587, .027522865530, -.031582039317,
                     .000553842201, .004777257511, -.001077301085};
               g.assign(f, sizeof(f) / sizeof(f[0]));
               }
               break;
            case 14:
               {
               const double f[] = {.077852054085, .396539319482, .729132090846,
                     .469782287405, -.143906003929, -.224036184994,
                     .071309219267, .080612609151, -.038029936935,
                     -.016574541631, .012550998556, .000429577973,
                     -.001801640704, .000353713800};
               g.assign(f, sizeof(f) / sizeof(f[0]));
               }
               break;
            case 16:
               {
               const double f[] = {.054415842243, .312871590914, .675630736297,
                     .585354683654, -.015829105256, -.284015542962,
                     .000472484574, .128747426620, -.017369301002,
                     -.044088253931, .013981027917, .008746094047,
                     -.004870352993, -.000391740373, .000675449406,
                     -.000117476784};
               g.assign(f, sizeof(f) / sizeof(f[0]));
               }
               break;
            case 18:
               {
               const double f[] = {.038077947364, .243834674613, .604823123690,
                     .657288078051, .133197385825, -.293273783279,
                     -.096840783223, .148540749338, .030725681479,
                     -.067632829061, .000250947115, .022361662124,
                     -.004723204758, -.004281503682, .001847646883,
                     .000230385764, -.000251963189, .000039347320};
               g.assign(f, sizeof(f) / sizeof(f[0]));
               }
               break;
            case 20:
               {
               const double f[] = {.026670057901, .188176800078, .527201188932,
                     .688459039454, .281172343661, -.249846424327,
                     -.195946274377, .127369340336, .093057364604,
                     -.071394147166, -.029457536822, .033212674059,
                     .003606553567, -.010733175483, .001395351747,
                     .001992405295, -.000685856695, -.000116466855,
                     .000093588670, -.000013264203};
               g.assign(f, sizeof(f) / sizeof(f[0]));
               }
               break;
               // Undefined
            default:
               cerr << "Undefined parameter (" << par << ") for wavelet type ("
                     << type << ")." << std::endl;
               return;
            }
         }
         break;
         // Symmlet
      case 4:
         {
         switch (par)
            {
            case 4:
               {
               const double f[] = {-.107148901418, -.041910965125,
                     .703739068656, 1.136658243408, .421234534204,
                     -.140317624179, -.017824701442, .045570345896};
               g.assign(f, sizeof(f) / sizeof(f[0]));
               }
               break;
            case 5:
               {
               const double f[] = {.038654795955, .041746864422,
                     -.055344186117, .281990696854, 1.023052966894,
                     .896581648380, .023478923136, -.247951362613,
                     -.029842499869, .027632152958};
               g.assign(f, sizeof(f) / sizeof(f[0]));
               }
               break;
            case 6:
               {
               const double f[] = {.021784700327, .004936612372,
                     -.166863215412, -.068323121587, .694457972958,
                     1.113892783926, .477904371333, -.102724969862,
                     -.029783751299, .063250562660, .002499922093,
                     -.011031867509};
               g.assign(f, sizeof(f) / sizeof(f[0]));
               }
               break;
            case 7:
               {
               const double f[] = {.003792658534, -.001481225915,
                     -.017870431651, .043155452582, .096014767936,
                     -.070078291222, .024665659489, .758162601964,
                     1.085782709814, .408183939725, -.198056706807,
                     -.152463871896, .005671342686, .014521394762};
               g.assign(f, sizeof(f) / sizeof(f[0]));
               }
               break;
            case 8:
               {
               const double f[] = {.002672793393, -.000428394300,
                     -.021145686528, .005386388754, .069490465911,
                     -.038493521263, -.073462508761, .515398670374,
                     1.099106630537, .680745347190, -.086653615406,
                     -.202648655286, .010758611751, .044823623042,
                     -.000766690896, -.004783458512};
               g.assign(f, sizeof(f) / sizeof(f[0]));
               }
               break;
            case 9:
               {
               const double f[] = {.001512487309, -.000669141509,
                     -.014515578553, .012528896242, .087791251554,
                     -.025786445930, -.270893783503, .049882830959,
                     .873048407349, 1.015259790832, .337658923602,
                     -.077172161097, .000825140929, .042744433602,
                     -.016303351226, -.018769396836, .000876502539,
                     .001981193736};
               g.assign(f, sizeof(f) / sizeof(f[0]));
               }
               break;
            case 10:
               {
               const double f[] = {.001089170447, .000135245020,
                     -.012220642630, -.002072363923, .064950924579,
                     .016418869426, -.225558972234, -.100240215031,
                     .667071338154, 1.088251530500, .542813011213,
                     -.050256540092, -.045240772218, .070703567550,
                     .008152816799, -.028786231926, -.001137535314,
                     .006495728375, .000080661204, -.000649589896};
               g.assign(f, sizeof(f) / sizeof(f[0]));
               }
               break;
               // Undefined
            default:
               cerr << "Undefined parameter (" << par << ") for wavelet type ("
                     << type << ")." << std::endl;
               return;
            }
         }
         break;
         // Vaidyanathan
      case 5:
         {
         const double f[] = {-.000062906118, .000343631905, -.000453956620,
               -.000944897136, .002843834547, .000708137504, -.008839103409,
               .003153847056, .019687215010, -.014853448005, -.035470398607,
               .038742619293, .055892523691, -.077709750902, -.083928884366,
               .131971661417, .135084227129, -.194450471766, -.263494802488,
               .201612161775, .635601059872, .572797793211, .250184129505,
               .045799334111};
         g.assign(f, sizeof(f) / sizeof(f[0]));
         }
         break;
         // Battle-Lemarie
      case 6:
         {
         switch (par)
            {
            case 1:
               {
               const double f[] = {0.578163, 0.280931, -0.0488618, -0.0367309,
                     0.012003, 0.00706442, -0.00274588, -0.00155701,
                     0.000652922, 0.000361781, -0.000158601, -0.0000867523};
               g.assign(f, sizeof(f) / sizeof(f[0]));
               }
               break;
            case 3:
               {
               const double f[] = {0.541736, 0.30683, -0.035498, -0.0778079,
                     0.0226846, 0.0297468, -0.0121455, -0.0127154, 0.00614143,
                     0.00579932, -0.00307863, -0.00274529, 0.00154624,
                     0.00133086, -0.000780468, -0.00065562, 0.000395946,
                     0.000326749, -0.000201818, -0.000164264, 0.000103307};
               g.assign(f, sizeof(f) / sizeof(f[0]));
               }
               break;
            case 5:
               {
               const double f[] = {0.528374, 0.312869, -0.0261771, -0.0914068,
                     0.0208414, 0.0433544, -0.0148537, -0.0229951, 0.00990635,
                     0.0128754, -0.00639886, -0.00746848, 0.00407882,
                     0.00444002, -0.00258816, -0.00268646, 0.00164132,
                     0.00164659, -0.00104207, -0.00101912, 0.000662836,
                     0.000635563, -0.000422485, -0.000398759, 0.000269842,
                     0.000251419, -0.000172685, -0.000159168, 0.000110709,
                     0.000101113};
               g.assign(f, sizeof(f) / sizeof(f[0]));
               }
               break;
               // Undefined
            default:
               cerr << "Undefined parameter (" << par << ") for wavelet type ("
                     << type << ")." << std::endl;
               return;
            }
         const vector<double> gcopy = g;
         const int len = g.size();
         g.init(2 * len - 1);
         for (int i = 0; i < len; i++)
            g(len - 1 + i) = g(len - 1 - i) = gcopy(i);
         }
         break;
         // Undefined
      default:
         cerr << "Undefined wavelet type (" << type << ")." << std::endl;
         return;
      }
   // normalise g and create quadrature filter
   g /= sqrt(g.sumsq());
   h = quadrature(g);
   // debug information
   trace << "wavelet initialised - type (" << type << ") par (" << par
         << ")." << std::endl;
   trace << "g = " << g << std::endl;
   trace << "h = " << h << std::endl;
   }

// informative / helper functions

int wavelet::getlimit(const int size, const int level) const
   {
   if (level <= 0)
      return 2;
   return std::max(2, size >> level);
   }

// transform / inverse functions - vector

void wavelet::transform(const vector<double>& in, vector<double>& out,
      const int level) const
   {
   assert(weight(in.size()) == 1);
   // resize the output vector if necessary
   out.init(in.size());
   // start at the largest heirarchy and work towards the smallest
   const int limit = getlimit(in.size(), level) << 1;
   for (int n = in.size(); n >= limit; n >>= 1)
      partial_transform(in, out, n);
   }

void wavelet::inverse(const vector<double>& in, vector<double>& out,
      const int level) const
   {
   assert(weight(in.size()) == 1);
   // resize the output vector if necessary
   out.init(in.size());
   // start at the smallest heirarchy and work towards the largest
   const int limit = getlimit(in.size(), level) << 1;
   for (int n = limit; n <= in.size(); n <<= 1)
      partial_inverse(in, out, n);
   }

// transform / inverse functions - matrix

void wavelet::transform(const matrix<double>& in, matrix<double>& out,
      const int level) const
   {
   assert(weight(in.size().rows()) == 1 && weight(in.size().cols()) == 1);
   // resize the output matrix if necessary
   out.init(in.size());
   // loop variables
   int i;
   // do the transform for each dimension
   vector<double> b;
   for (i = 0; i < in.size().rows(); i++)
      {
      in.extractrow(b, i);
      transform(b, b, level);
      out.insertrow(b, i);
      }
   for (i = 0; i < in.size().cols(); i++)
      {
      out.extractcol(b, i);
      transform(b, b, level);
      out.insertcol(b, i);
      }
   }

void wavelet::inverse(const matrix<double>& in, matrix<double>& out,
      const int level) const
   {
   assert(weight(in.size().rows()) == 1 && weight(in.size().cols()) == 1);
   // resize the output matrix if necessary
   out.init(in.size());
   // loop variables
   int i;
   // do the transform for each dimension
   vector<double> b;
   for (i = 0; i < in.size().rows(); i++)
      {
      in.extractrow(b, i);
      inverse(b, b, level);
      out.insertrow(b, i);
      }
   for (i = 0; i < in.size().cols(); i++)
      {
      out.extractcol(b, i);
      inverse(b, b, level);
      out.insertcol(b, i);
      }
   }

} // end namespace
