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

#include "stegosystem.h"

#include "rvstatistics.h"
#include "randgen.h"
#include "bitfield.h"
#include "fbstream.h"
#include <cstring>

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/erf.hpp>

namespace libcomm {

using libbase::trace;
using libbase::vector;
using libbase::matrix;
using libbase::randgen;

// Piece-wise Linear Modulator
double stegosystem::plmod(const double u)
   {
   if (u < 0.5)
      return u + 0.5;
   else if (u > 0.5)
      return u - 0.5;
   else
      return 0;
   }

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

stegosystem::stegosystem()
   {
   m_pCodec = NULL;
   }

stegosystem::~stegosystem()
   {
   FreeErrorControl();
   }

/////////////////////////////////////////////////////////////////////////////
// informative functions

// returns raw data size available, in bits (after variable-density interleaver only)
int stegosystem::GetRawSize(double dInterleaverDensity) const
   {
   return int(floor(GetImagePixels() * dInterleaverDensity));
   }

// returns the length of each encoded vector (in bits)
int stegosystem::GetOutputSize() const
   {
   return int(round((m_pCodec == NULL) ? 1 : m_pCodec->output_bits()));
   }

// returns the length of each uncoded vector (in bits)
int stegosystem::GetInputSize() const
   {
   return int(round((m_pCodec == NULL) ? 1 : m_pCodec->input_bits()));
   }

// returns the length of the data vector (in elements)
int stegosystem::GetDataSize(double dInterleaverDensity, int nEmbedRate) const
   {
   const int nBlocks = (GetRawSize(dInterleaverDensity) / nEmbedRate)
         / GetOutputSize();
   return nBlocks * GetInputSize() / GetDataWidth();
   }

// returns the width of each data element as passed to the encoder (in bits)
int stegosystem::GetDataWidth() const
   {
   return (m_pCodec == NULL) ? 1 : int(round(log2(m_pCodec->num_inputs())));
   }

// returns the code rate
double stegosystem::GetCodeRate() const
   {
   return double(GetInputSize()) / double(GetOutputSize());
   }

/////////////////////////////////////////////////////////////////////////////
//

void stegosystem::LoadErrorControl(const char* sCodec)
   {
   // load encoder and puncturing pattern
   FreeErrorControl();
   if (strlen(sCodec) != 0)
      {
      std::ifstream file(sCodec, std::ios_base::in | std::ios_base::binary);
      file >> m_pCodec;
      }
   }

void stegosystem::FreeErrorControl()
   {
   if (m_pCodec != NULL)
      {
      delete m_pCodec;
      m_pCodec = NULL;
      }
   }

void stegosystem::LoadDataFile(const char* sPathName, vector<int>& d, int n)
   {
   assert(strlen(sPathName) != 0);
   libbase::bitfield b(n);
   libbase::ifbstream file(sPathName);
   for (int i = 0; i < d.size() && !(file.eof() && file.buffer_bits() == 0); i++)
      {
      file >> b;
      d(i) = b;
      }
   file.close();
   }

void stegosystem::EncodeData(const vector<int>& d, vector<int>& e)
   {
   assert(m_pCodec != NULL);
   // get block size data
   const int tau = m_pCodec->output_block_size();
   const int m = m_pCodec->tail_length();
   // set up source / encoded vectors
   vector<int> source(tau), encoded;
   // BPSK blockmodem & modulated signal
   mpsk mdm(2);
   vector<sigspace> signal;
   // loop for all blocks
   int k = 0;
   for (int j = 0; j < d.size(); j += (tau - m))
      {
      // keep user happy
      DisplayProgress(j, d.size(), 3, 5);
      // temporary variable
      int i;
      // build source block
      for (i = 0; i < tau - m; i++)
         source(i) = d(j + i);
      for (i = tau - m; i < tau; i++)
         source(i) = fsm::tail;
      // encode
      m_pCodec->encode(source, encoded);
      // modulate
      mdm.modulate(m_pCodec->num_outputs(), encoded, signal);
      // write into output stream
      for (i = 0; i < signal.size(); i++)
         e(k++) = mdm.demodulate(signal(i));
      }
   // fill in the rest with zeros
   while (k < e.size())
      e(k++) = 0;
   }

void stegosystem::DecodeData(double dInterleaverDensity, int nEmbedRate,
      const double dSNR, const vector<sigspace>& s, vector<int>& d)
   {
   assert(m_pCodec != NULL);
   // initialize data vector
   d.init(GetDataSize(dInterleaverDensity, nEmbedRate));
   // get block size data
   const int tau = m_pCodec->output_block_size();
   const int m = m_pCodec->tail_length();
   // set up signal, decoded vectors and probability matrix
   vector<sigspace> signal(GetOutputSize());
   trace << "Signal space block size = " << signal.size() << std::endl;
   vector<int> decoded;
   vector<vector<double> > ptable;
   // BPSK blockmodem
   mpsk mdm(2);
   // set up channel
   laplacian<sigspace> chan;
   chan.set_eb(mdm.bit_energy());
   chan.set_parameter(dSNR);
   // loop for all blocks
   int k = 0;
   for (int j = 0; j < d.size(); j += (tau - m))
      {
      // keep user happy
      DisplayProgress(j, d.size(), 4, 5);
      // temporary variable
      int i;
      // build signal block
      for (i = 0; i < signal.size(); i++)
         signal(i) = s(k++);
      // demodulate (build probability table)
      mdm.demodulate(chan, signal, ptable);
      // decode
      m_pCodec->init_decoder(ptable);
      for (i = 0; i < m_pCodec->num_iter(); i++)
         m_pCodec->decode(decoded);
      // write into output stream
      for (i = 0; i < tau - m; i++)
         d(j + i) = decoded(i);
      }
   }

void stegosystem::DemodulateData(const vector<sigspace>& s, vector<int>& d)
   {
   assert(s.size() > 0);
   d.init(s.size());
   // BPSK blockmodem
   mpsk mdm(2);
   // demodulate signal
   for (int i = 0; i < s.size(); i++)
      d(i) = mdm.demodulate(s(i));
   }

double stegosystem::EstimateSNR(const double dRate, const vector<sigspace>& rx,
      const vector<sigspace>& tx, double* pdSNRreal)
   {
   assert(rx.size() == tx.size());
   libbase::rvstatistics r1, r2;
   for (int i = 0; i < rx.size(); i++)
      {
      // get in-phase component
      const double d = rx(i).i();
      // use only outer side-lobes for first estimate
      if (d <= -1)
         r1.insert(d + 1);
      else if (d >= +1)
         r1.insert(d - 1);
      // use knowledge of what we embedded for second estimate:
      r2.insert(d - tx(i).i());
      }
   const double dLambda1 = sqrt(r1.var() + r1.mean() * r1.mean()) / sqrt(2.0);
   const double dLambda2 = r2.sigma() / sqrt(2.0);
   const double dSNRest1 = -20
         * log10(dLambda1 * sqrt(double(2)) * sqrt(dRate));
   const double dSNRest2 = -20
         * log10(dLambda2 * sqrt(double(2)) * sqrt(dRate));
   trace << "Channel estimate 1: mean = " << r1.mean() << ", sigma = "
         << r1.sigma() << ", lambda = " << dLambda1 << ", SNR = " << dSNRest1
         << "dB" << std::endl;
   trace << "Channel estimate 2: mean = " << r2.mean() << ", sigma = "
         << r2.sigma() << ", lambda = " << dLambda2 << ", SNR = " << dSNRest2
         << "dB" << std::endl;
   if (pdSNRreal != NULL)
      *pdSNRreal = dSNRest2;
   return dSNRest1;
   }

double stegosystem::ComputeChiSquare(const vector<sigspace>& rx, const vector<
      sigspace>& tx, int nBins, double dSNR)
   {
   assert(rx.size() == tx.size());
   // create error vector
   const int N = rx.size();
   vector<double> e(N);
      {
      for (int i = 0; i < e.size(); i++)
         e(i) = rx(i).i() - tx(i).i();
      }
   // compute histogram of error with given number of bins
   // TODO: refactor to make use of histogram class
   vector<int> h(nBins);
   double dMin = e.min();
   double dMax = e.max();
   double dStep = (dMax - dMin) / double(nBins);
   trace << "Computing histogram with " << nBins << " bins in [" << dMin
         << ", " << dMax << "]" << std::endl;
   dMin += dStep / 2;
   dMax -= dStep / 2;
      {
      h = 0;
      for (int i = 0; i < N; i++)
         h(libbase::limit(int(round((e(i) - dMin) / dStep)), 0, nBins - 1))++;
      }
   // compute chi-square metric, assuming a laplacian distribution for the given SNR
   const double lambda = pow(10.0, -dSNR / 20.0) / sqrt(2.0);
   double chisq = 0;
   for (int i = 0; i < nBins; i++)
      {
      const double ni = N * dStep * 1 / (2 * lambda) * exp(-fabs(dMin + i
            * dStep) / lambda);
      const double d = h(i) - ni;
      trace << "Bin " << i << ": Ni=" << h(i) << ", ni=" << ni << std::endl;
      chisq += d * d / ni;
      }
   // compute probability of null hypothesis for that chi-square metric
#ifndef NDEBUG
   const double p = 1.0 - boost::math::gamma_p(0.5 * (nBins - 1), 0.5 * chisq);
   trace << "Probability of null hypothesis = " << p << ", given ChiSq = "
         << chisq << std::endl;
#endif
   return chisq;
   }

void stegosystem::GenerateSourceSequence(vector<int>& d, int n, int seed)
   {
   assert(d.size() > 0);
   randgen r;
   r.seed(seed);
   for (int i = 0; i < d.size(); i++)
      d(i) = r.ival(1 << n);
   }

void stegosystem::GenerateEmbedSequence(vector<double>& u, int seed)
   {
   assert(u.size() > 0);
   randgen r;
   r.seed(seed);
   for (int i = 0; i < u.size(); i++)
      u(i) = r.fval_closed();
   }

void stegosystem::DemodulateEmbedSequence(const vector<double>& v,
      const vector<double>& u, vector<sigspace>& s)
   {
   assert(u.size() > 0);
   assert(u.size() == v.size());
   s.init(u.size());
   mpsk mdm(2);
   for (int i = 0; i < u.size(); i++)
      {
      const double d = (v(i) - u(i)) / (plmod(u(i)) - u(i));
      s(i) = mdm[0] + d * (mdm[1] - mdm[0]);
      }
   }

void stegosystem::ModulateEmbedSequence(const vector<int>& d, const vector<
      double>& u, vector<double>& v)
   {
   assert(u.size() > 0);
   assert(u.size() == d.size());
   v.init(u.size());
   for (int i = 0; i < u.size(); i++)
      v(i) = d(i) ? plmod(u(i)) : u(i);
   }

void stegosystem::ConvertToUniform(const vector<double>& g, vector<double>& v)
   {
   assert(g.size() > 0);
   v.init(g.size());
   for (int i = 0; i < g.size(); i++)
      {
      if ((i & 0xff) == 0)
         DisplayProgress(i, g.size(), 2, (m_pCodec == NULL) ? 3 : 5);
      v(i) = (boost::math::erf(g(i) / sqrt(double(2))) + 1.0) / 2.0;
      }
   }

void stegosystem::ConvertToGaussian(const vector<double>& v, vector<double>& g)
   {
   assert(v.size() > 0);
   g.init(v.size());
   for (int i = 0; i < v.size(); i++)
      {
      if ((i & 0xff) == 0)
         DisplayProgress(i, v.size(), 1, 4);
      g(i) = boost::math::erf_inv(2 * v(i) - 1) * sqrt(double(2));
      }
   }

void stegosystem::NormalizeGaussian(vector<double>& g, bool bPresetStrength,
      double dEmbedStrength)
   {
   const double dMeanEst = g.mean();
   const double dSigmaEst = g.sigma();
   trace << "Embedding estimate: mean = " << dMeanEst << ", strength = " << 20
         * log10(dSigmaEst) << "dB" << std::endl;
   if (bPresetStrength)
      {
      const double scale = pow(10.0, dEmbedStrength / 20);
      g -= dMeanEst;
      g /= scale;
      }
   else
      {
      g -= dMeanEst;
      g /= dSigmaEst;
      }
   }

void stegosystem::GenerateInterleaver(vector<int>& v, int in, int out, int seed)
   {
   v.init(in);
   v = -1;
   randgen r;
   r.seed(seed);
   for (int i = 0; i < out; i++)
      {
      if ((i & 0xff) == 0)
         DisplayProgress(i, out, 1, (m_pCodec == NULL) ? 3 : 5);
      int index;
      do
         {
         index = r.ival(in);
         } while (v(index) >= 0);
      v(index) = i;
      }
   }

void stegosystem::DeInterleaveMessage(const vector<int>& viIndex, const vector<
      double>& vdIn, vector<double>& vdOut)
   {
   vdOut.init(viIndex.max() + 1);
   for (int i = 0; i < vdIn.size(); i++)
      if (viIndex(i) >= 0)
         vdOut(viIndex(i)) = vdIn(i);
   }

void stegosystem::InterleaveMessage(const vector<int>& viIndex, const vector<
      double>& vdIn, vector<double>& vdOut)
   {
   vdOut.init(viIndex.size());
   vdOut = 0;
   for (int i = 0; i < vdOut.size(); i++)
      if (viIndex(i) >= 0)
         vdOut(i) = vdIn(viIndex(i));
   }

void stegosystem::BandwidthCompressor(int nRate, const vector<sigspace>& viIn,
      vector<sigspace>& viOut)
   {
   viOut.init(viIn.size() / nRate);
   for (int i = 0; i < viOut.size(); i++)
      {
      sigspace s(0, 0);
      for (int j = 0; j < nRate; j++)
         s += viIn(i * nRate + j);
      s /= double(nRate);
      viOut(i) = s;
      }
   }

void stegosystem::BandwidthExpander(int nRate, const vector<int>& viIn, vector<
      int>& viOut)
   {
   for (int i = 0; i < viIn.size(); i++)
      for (int j = 0; j < nRate; j++)
         viOut(i * nRate + j) = viIn(i);
   for (int k = viIn.size() * nRate; k < viOut.size(); k++)
      viOut(k) = 0;
   }

void stegosystem::BandwidthExpander(int nRate, const vector<double>& viIn,
      vector<double>& viOut)
   {
   for (int i = 0; i < viIn.size(); i++)
      for (int j = 0; j < nRate; j++)
         viOut(i * nRate + j) = viIn(i);
   for (int k = viIn.size() * nRate; k < viOut.size(); k++)
      viOut(k) = 0.5;
   }

void stegosystem::BandwidthExpander(int nRate, const vector<sigspace>& viIn,
      vector<sigspace>& viOut)
   {
   for (int i = 0; i < viIn.size(); i++)
      for (int j = 0; j < nRate; j++)
         viOut(i * nRate + j) = viIn(i);
   for (int k = viIn.size() * nRate; k < viOut.size(); k++)
      viOut(k) = sigspace(0, 0);
   }

} // end namespace
