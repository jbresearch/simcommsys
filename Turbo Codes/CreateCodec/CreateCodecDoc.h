// CreateCodecDoc.h : interface of the CCreateCodecDoc class
//
/////////////////////////////////////////////////////////////////////////////

#if !defined(AFX_CREATECODECDOC_H__128DB58B_11E1_4AC9_8440_0BC6739522C5__INCLUDED_)
#define AFX_CREATECODECDOC_H__128DB58B_11E1_4AC9_8440_0BC6739522C5__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "vector.h"
#include "matrix.h"
#include "bitfield.h"
#include "interleaver.h"
#include "codec.h"

class CCreateCodecDoc : public CDocument
{
protected: // create from serialization only
        CCreateCodecDoc();
        DECLARE_DYNCREATE(CCreateCodecDoc)

// Attributes
public:

// Operations
public:

// Overrides
        // ClassWizard generated virtual function overrides
        //{{AFX_VIRTUAL(CCreateCodecDoc)
        public:
        virtual BOOL OnNewDocument();
        virtual void Serialize(CArchive& ar);
        //}}AFX_VIRTUAL

// Implementation
public:
   libcomm::codec* CreateCodec();
        CString GetStringBool(const CString& sName);
        CString GetStringInt(const CString& sName);
   CString GetStringInterleaver(const libcomm::interleaver *pInterleaver);
        CString GetStringGenerator();
        CString GetStringEncoder();
        CString GetStringCodec();
        void ResizeGenerator(const int nInputs, const int nOutputs);
        void AddInterleaver(libcomm::interleaver *pInterleaver);
        void DeleteInterleavers();
        void SetIntValue(const CString& sName, const int nValue);
        void SetBoolValue(const CString& sName, const bool bValue);
   void SetGenerator(const libbase::matrix<libbase::bitfield>& mbGenerator);
        void SetArithmetic(const int nType);
        void SetCodecType(const int nType);
        void SetEncoderType(const int nType);
        int GetIntValue(const CString& sName) const;
        bool GetBoolValue(const CString& sName) const;
   libbase::matrix<libbase::bitfield> GetGenerator() const { return m_mbGenerator; };
   int GetArithmetic() const { return m_nArithmetic; };
   int GetCodecType() const { return m_nCodecType; };
   int GetEncoderType() const { return m_nEncoderType; };
        virtual ~CCreateCodecDoc();
#ifdef _DEBUG
        virtual void AssertValid() const;
        virtual void Dump(CDumpContext& dc) const;
#endif

protected:
        int m_nCodecType, m_nArithmetic;
   int m_nTau, m_nIterations;
   bool m_bSimile, m_bTerminated, m_bParallel;
   int m_nEncoderType;
   libbase::matrix<libbase::bitfield> m_mbGenerator;
   libbase::vector<libcomm::interleaver *> m_vpInterleavers;

// Generated message map functions
protected:
        //{{AFX_MSG(CCreateCodecDoc)
                // NOTE - the ClassWizard will add and remove member functions here.
                //    DO NOT EDIT what you see in these blocks of generated code !
        //}}AFX_MSG
        DECLARE_MESSAGE_MAP()
};

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_CREATECODECDOC_H__128DB58B_11E1_4AC9_8440_0BC6739522C5__INCLUDED_)
