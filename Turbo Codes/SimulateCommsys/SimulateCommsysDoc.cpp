// SimulateCommsysDoc.cpp : implementation of the CSimulateCommsysDoc class
//

#include "stdafx.h"
#include "SimulateCommsys.h"

#include "SimulateCommsysDoc.h"
#include "ArchiveStreamBuf.h"

#include "timer.h"
#include "mpsk.h"
#include "awgn.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CSimulateCommsysDoc

IMPLEMENT_DYNCREATE(CSimulateCommsysDoc, CDocument)

BEGIN_MESSAGE_MAP(CSimulateCommsysDoc, CDocument)
//{{AFX_MSG_MAP(CSimulateCommsysDoc)
// NOTE - the ClassWizard will add and remove mapping macros here.
//    DO NOT EDIT what you see in these blocks of generated code!
//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CSimulateCommsysDoc construction/destruction

CSimulateCommsysDoc::CSimulateCommsysDoc()
   {
   // TODO: add one-time construction code here
   m_pChannel = NULL;
   m_pModulator = NULL;
   m_pPuncture = NULL;
   m_pCodec = NULL;
   }

CSimulateCommsysDoc::~CSimulateCommsysDoc()
   {
   Free();
   }

BOOL CSimulateCommsysDoc::OnNewDocument()
   {
   if (!CDocument::OnNewDocument())
      return FALSE;
   
   // Note: SDI documents reuse this document
   Free();
   m_pChannel = new libcomm::awgn;
   m_pModulator = new libcomm::mpsk(2);
   m_pPuncture = NULL;
   m_pCodec = NULL;

   m_dAccuracy = 0.15;
   m_dConfidence = 0.90;

   ClearResults();
   
   return TRUE;
   }


/////////////////////////////////////////////////////////////////////////////
// CSimulateCommsysDoc serialization

void CSimulateCommsysDoc::Serialize(CArchive& ar)
   {
   if(ar.IsStoring())
      {
      // attach a stream to archive object
      libwin::CArchiveStreamBuf buf(&ar);
      std::ostream file(&buf);

      // Print information on the statistical accuracy of results being worked
      file << "#% Date: " << libbase::timer::date() << "\n";
      file << "#\n";
      file << "#% Tolerance: " << 100*m_dAccuracy << "%\n";
      file << "#% Confidence: " << 100*m_dConfidence << "%\n";
      file << "#% Codec: " << m_pCodec->description() << "\n";
      file << "#% Modulation: " << m_pModulator->description() << "\n";
      if(m_pPuncture != NULL)
         file << "#% Puncturing: " << m_pPuncture->description() << "\n";
      file << "#% Channel: " << m_pChannel->description() << "\n";
      file << "#\n";
      
      // Print the results
      file.precision(6);
      for(std::list<SResult>::iterator p=m_lsResults.begin(); p != m_lsResults.end(); p++)
         {
         file << "# Time taken: " << libbase::timer::format(p->dElapsed) << "\n";
         file << p->dSNR << "\t";
         for(int i=0; i<p->vdEstimate.size(); i++)
            {
            file << p->vdEstimate(i) << "\t";
            file << p->vdEstimate(i) * p->vdError(i) << "\t";
            }
         file << p->iSamples << "\n";
         }
      }
   else
      {
      // TODO: add loading code here
      }
   }

/////////////////////////////////////////////////////////////////////////////
// CSimulateCommsysDoc diagnostics

#ifdef _DEBUG
void CSimulateCommsysDoc::AssertValid() const
   {
   CDocument::AssertValid();
   }

void CSimulateCommsysDoc::Dump(CDumpContext& dc) const
   {
   CDocument::Dump(dc);
   }
#endif //_DEBUG

/////////////////////////////////////////////////////////////////////////////
// CSimulateCommsysDoc commands

void CSimulateCommsysDoc::Free()
   {
   if(m_pChannel != NULL)
      delete m_pChannel;
   if(m_pModulator != NULL)
      delete m_pModulator;
   if(m_pPuncture != NULL)
      delete m_pPuncture;
   if(m_pCodec != NULL)
      delete m_pCodec;
   }

// public settings functions

// system components

void CSimulateCommsysDoc::SetChannel(libcomm::channel *pChannel)
   {
   ASSERT(!ResultsPresent());
   if(m_pChannel != NULL)
      delete m_pChannel;
   m_pChannel = pChannel;
   }

void CSimulateCommsysDoc::SetModulator(libcomm::modulator *pModulator)
   {
   ASSERT(!ResultsPresent());
   if(m_pModulator != NULL)
      delete m_pModulator;
   m_pModulator = pModulator;
   }

void CSimulateCommsysDoc::SetPuncture(libcomm::puncture *pPuncture)
   {
   ASSERT(!ResultsPresent());
   if(m_pPuncture != NULL)
      delete m_pPuncture;
   m_pPuncture = pPuncture;
   }

void CSimulateCommsysDoc::SetCodec(libcomm::codec *pCodec)
   {
   ASSERT(!ResultsPresent());
   if(m_pCodec != NULL)
      delete m_pCodec;
   m_pCodec = pCodec;
   }

// simulation settings

void CSimulateCommsysDoc::SetAccuracy(const double dValue)
   {
   ASSERT(!ResultsPresent());
   ASSERT(dValue > 0.0 && dValue < 1.0);
   m_dAccuracy = dValue;
   }

void CSimulateCommsysDoc::SetConfidence(const double dValue)
   {
   ASSERT(!ResultsPresent());
   ASSERT(dValue > 0.0 && dValue < 1.0);
   m_dConfidence = dValue;
   }

// results

void CSimulateCommsysDoc::ClearResults()
   {
   m_lsResults.clear();
   }

void CSimulateCommsysDoc::InsertResults(const double dSNR, const int iSamples, const libbase::vector<double>& vdEstimate, const libbase::vector<double>& vdError, const double dElapsed)
   {
   SResult result;
   result.dSNR = dSNR;
   result.iSamples = iSamples;
   result.vdEstimate = vdEstimate;
   result.vdError = vdError;
   result.dElapsed = dElapsed;
   m_lsResults.push_back(result);
   SetModifiedFlag();
   }
