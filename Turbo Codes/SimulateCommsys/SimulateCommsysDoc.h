// SimulateCommsysDoc.h : interface of the CSimulateCommsysDoc class
//
/////////////////////////////////////////////////////////////////////////////

#if !defined(AFX_SIMULATECOMMSYSDOC_H__B3DF2C69_5432_4BBA_A7CE_BE51F7D2C08B__INCLUDED_)
#define AFX_SIMULATECOMMSYSDOC_H__B3DF2C69_5432_4BBA_A7CE_BE51F7D2C08B__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "channel.h"
#include "modulator.h"
#include "puncture.h"
#include "codec.h"
#include "timer.h"
#include <list>

class CSimulateCommsysDoc : public CDocument
{
protected: // create from serialization only
        CSimulateCommsysDoc();
        DECLARE_DYNCREATE(CSimulateCommsysDoc)

// Attributes
public:

// Operations
public:

// Overrides
        // ClassWizard generated virtual function overrides
        //{{AFX_VIRTUAL(CSimulateCommsysDoc)
        public:
        virtual BOOL OnNewDocument();
        virtual void Serialize(CArchive& ar);
        //}}AFX_VIRTUAL

// Implementation
public:
   // system components
   void SetCodec(libcomm::codec *pCodec);
   void SetPuncture(libcomm::puncture *pPuncture);
   void SetModulator(libcomm::modulator<libcomm::sigspace> *pModulator);
   void SetChannel(libcomm::channel<libcomm::sigspace> *pChannel);
   libcomm::codec *GetCodec() const { return m_pCodec; };
   libcomm::puncture *GetPuncture() const { return m_pPuncture; };
   libcomm::modulator<libcomm::sigspace> *GetModulator() const { return m_pModulator; };
   libcomm::channel<libcomm::sigspace> *GetChannel() const { return m_pChannel; };
   // simulation settings
   void SetAccuracy(const double dValue);
   void SetConfidence(const double dValue);
   double GetAccuracy() const { return m_dAccuracy; };
   double GetConfidence() const { return m_dConfidence; };
   // results
   void ClearResults();
   void InsertResults(const double dSNR, const int iSamples, const libbase::vector<double>& vdEstimate, const libbase::vector<double>& vdError, const double dElapsed);
   bool ResultsPresent() const { return !m_lsResults.empty(); };
   // destructor
        virtual ~CSimulateCommsysDoc();
#ifdef _DEBUG
        virtual void AssertValid() const;
        virtual void Dump(CDumpContext& dc) const;
#endif

protected:
   // system components
   libcomm::channel<libcomm::sigspace> *m_pChannel;
   libcomm::modulator<libcomm::sigspace> *m_pModulator;
   libcomm::puncture *m_pPuncture;
   libcomm::codec *m_pCodec;
   // simulation settings
   double m_dAccuracy, m_dConfidence;
   // results
   typedef struct SResult {
      double   dSNR;
      int      iSamples;
      libbase::vector<double> vdEstimate, vdError;
      double   dElapsed;
      };
   std::list<SResult> m_lsResults;

   // internal functions
   void Free();

// Generated message map functions
protected:
        //{{AFX_MSG(CSimulateCommsysDoc)
                // NOTE - the ClassWizard will add and remove member functions here.
                //    DO NOT EDIT what you see in these blocks of generated code !
        //}}AFX_MSG
        DECLARE_MESSAGE_MAP()
};

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_SIMULATECOMMSYSDOC_H__B3DF2C69_5432_4BBA_A7CE_BE51F7D2C08B__INCLUDED_)
