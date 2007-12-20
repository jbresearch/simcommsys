// CreateCodecDoc.cpp : implementation of the CCreateCodecDoc class
//

#include "stdafx.h"
#include "CreateCodec.h"
#include "CreateCodecDoc.h"
#include "ArchiveStreamBuf.h"

#include "serializer_libcomm.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CCreateCodecDoc

IMPLEMENT_DYNCREATE(CCreateCodecDoc, CDocument)

BEGIN_MESSAGE_MAP(CCreateCodecDoc, CDocument)
//{{AFX_MSG_MAP(CCreateCodecDoc)
// NOTE - the ClassWizard will add and remove mapping macros here.
//    DO NOT EDIT what you see in these blocks of generated code!
//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CCreateCodecDoc construction/destruction

CCreateCodecDoc::CCreateCodecDoc()
   {
   // TODO: add one-time construction code here
   m_mbGenerator.init(1,1);
   m_vpInterleavers.init(0);
   }

CCreateCodecDoc::~CCreateCodecDoc()
   {
   DeleteInterleavers();
   }

BOOL CCreateCodecDoc::OnNewDocument()
   {
   if (!CDocument::OnNewDocument())
      return FALSE;

   // TODO: add reinitialization code here
   // (SDI documents will reuse this document)
   m_nArithmetic = -1;
   m_nCodecType = -1;
   m_nTau = 1024;
   m_nIterations = 10;
   m_bSimile = false;
   m_bTerminated = false;
   m_bParallel = false;
   m_nEncoderType = -1;
   m_mbGenerator.init(1,1);
   DeleteInterleavers();

   return TRUE;
   }



/////////////////////////////////////////////////////////////////////////////
// CCreateCodecDoc serialization

void CCreateCodecDoc::Serialize(CArchive& ar)
   {
   if (ar.IsStoring())
      {
      // attach a stream to archive object
      libwin::CArchiveStreamBuf buf(&ar);
      std::ostream sout(&buf);
      // create & write codec
      libcomm::codec *cdc = CreateCodec();
      sout << cdc;
      delete cdc;
      }
   else
      {
      // TODO: add loading code here
      }
   }

/////////////////////////////////////////////////////////////////////////////
// CCreateCodecDoc diagnostics

#ifdef _DEBUG
void CCreateCodecDoc::AssertValid() const
   {
   CDocument::AssertValid();
   }

void CCreateCodecDoc::Dump(CDumpContext& dc) const
   {
   CDocument::Dump(dc);
   }
#endif //_DEBUG

/////////////////////////////////////////////////////////////////////////////
// CCreateCodecDoc commands

void CCreateCodecDoc::SetCodecType(const int nType)
   {
   ASSERT(nType >= 0 && nType <= 3);

   m_nCodecType = nType;
   if(m_nCodecType <= 0)
      m_nArithmetic = -1;
   else if(m_nArithmetic < 0)
      m_nArithmetic = 0;
   }

void CCreateCodecDoc::SetArithmetic(const int nType)
   {
   if(m_nCodecType > 0)
      m_nArithmetic = nType;
   }

void CCreateCodecDoc::SetEncoderType(const int nType)
   {
   ASSERT(nType >= 0 && nType <= 1);

   m_nEncoderType = nType;
   }

void CCreateCodecDoc::SetGenerator(const libbase::matrix<libbase::bitfield>& mbGenerator)
   {
   ASSERT(mbGenerator.xsize() == m_mbGenerator.xsize());
   ASSERT(mbGenerator.ysize() == m_mbGenerator.ysize());
   m_mbGenerator = mbGenerator;
   }

int CCreateCodecDoc::GetIntValue(const CString &sName) const
   {
   if(sName.CompareNoCase("iterations") == 0)
      return m_nIterations;
   if(sName.CompareNoCase("block size") == 0)
      return m_nTau;
   if(sName.CompareNoCase("inputs") == 0)
      return m_mbGenerator.xsize();
   if(sName.CompareNoCase("outputs") == 0)
      return m_mbGenerator.ysize();
   MessageBox(NULL, "Unkown integer name \""+sName+"\"", NULL, MB_OK | MB_ICONERROR);
   return 0;
   }

void CCreateCodecDoc::SetIntValue(const CString &sName, const int nValue)
   {
   if(sName.CompareNoCase("iterations") == 0)
      m_nIterations = nValue;
   else if(sName.CompareNoCase("block size") == 0)
      m_nTau = nValue;
   else if(sName.CompareNoCase("inputs") == 0)
      ResizeGenerator(nValue, m_mbGenerator.ysize());
   else if(sName.CompareNoCase("outputs") == 0)
      ResizeGenerator(m_mbGenerator.xsize(), nValue);
   else
      MessageBox(NULL, "Unkown integer name \""+sName+"\"", NULL, MB_OK | MB_ICONERROR);
   }

bool CCreateCodecDoc::GetBoolValue(const CString &sName) const
   {
   if(sName.CompareNoCase("parallel") == 0)
      return m_bParallel;
   if(sName.CompareNoCase("simile") == 0)
      return m_bSimile;
   if(sName.CompareNoCase("terminated") == 0)
      return m_bTerminated;
   MessageBox(NULL, "Unkown bool name \""+sName+"\"", NULL, MB_OK | MB_ICONERROR);
   return false;
   }

void CCreateCodecDoc::SetBoolValue(const CString &sName, const bool bValue)
   {
   if(sName.CompareNoCase("parallel") == 0)
      m_bParallel = bValue;
   else if(sName.CompareNoCase("simile") == 0)
      m_bSimile = bValue;
   else if(sName.CompareNoCase("terminated") == 0)
      m_bTerminated = bValue;
   else
      MessageBox(NULL, "Unkown bool name \""+sName+"\"", NULL, MB_OK | MB_ICONERROR);
   }

void CCreateCodecDoc::DeleteInterleavers()
   {
   for(int i=0; i<m_vpInterleavers.size(); i++)
      delete m_vpInterleavers(i);
   m_vpInterleavers.init(0);
   }

void CCreateCodecDoc::AddInterleaver(libcomm::interleaver *pInterleaver)
   {
   libbase::vector<libcomm::interleaver *> vpInterleavers(m_vpInterleavers);
   const int nSize = m_vpInterleavers.size();
   m_vpInterleavers.init(nSize+1);
   for(int i=0; i<nSize; i++)
      m_vpInterleavers(i) = vpInterleavers(i);
   m_vpInterleavers(nSize) = pInterleaver;
   }

void CCreateCodecDoc::ResizeGenerator(const int nInputs, const int nOutputs)
   {
   libbase::matrix<libbase::bitfield> mbGenerator(m_mbGenerator);
   const int nXSize = m_mbGenerator.xsize();
   const int nYSize = m_mbGenerator.ysize();
   m_mbGenerator.init(nInputs, nOutputs);
   m_mbGenerator = libbase::bitfield("");
   for(int i=0; i<min(nXSize, nInputs); i++)
      for(int j=0; j<min(nYSize, nOutputs); j++)
         m_mbGenerator(i,j) = mbGenerator(i,j);
   }

CString CCreateCodecDoc::GetStringCodec()
   {
   CString sMath, sName;
   switch(m_nArithmetic)
      {
      case 0:
         sMath = "<logreal>";
         break;
      case 1:
         sMath = "<logrealfast>";
         break;
      case 2:
         sMath = "<mpreal>";
         break;
      case 3:
         sMath = "<mpgnu>";
         break;
      default:
         sMath = "<unknown>";
         break;
      }
   switch(m_nCodecType)
      {
      case -1:
         sName = "<empty>";
         break;
      case 0:
         sName = "uncoded";
         break;
      case 1:
         sName = "mapcc"+sMath;
         break;
      case 2:
         sName = "turbo"+sMath;
         break;
      case 3:
         sName = "diffturbo"+sMath;
         break;
      default:
         sName = "unknown"+sMath;
         break;
      }
   return "codec: "+sName;
   }

CString CCreateCodecDoc::GetStringEncoder()
   {
   CString sName;
   switch(m_nEncoderType)
      {
      case -1:
         sName = "<empty>";
         break;
      case 0:
         sName = "nrcc";
         break;
      case 1:
         sName = "rscc";
         break;
      default:
         sName = "unknown";
         break;
      }
   return "encoder: "+sName;
   }

CString CCreateCodecDoc::GetStringGenerator()
   {
   CString sTemp;
   sTemp.Format("generator: %dx%d", m_mbGenerator.xsize(), m_mbGenerator.ysize());
   return sTemp;
   }

CString CCreateCodecDoc::GetStringInterleaver(const libcomm::interleaver *pInterleaver)
   {
   return pInterleaver->description().c_str();
   }

CString CCreateCodecDoc::GetStringInt(const CString &sName)
   {
   CString sTemp;
   sTemp.Format("%s: %d", sName, GetIntValue(sName));
   return sTemp;
   }

CString CCreateCodecDoc::GetStringBool(const CString &sName)
   {
   CString sTemp;
   sTemp.Format("%s: %s", sName, GetBoolValue(sName) ? "yes" : "no");
   return sTemp;
   }

libcomm::codec* CCreateCodecDoc::CreateCodec()
   {
   // create encoder
   libcomm::fsm *encoder;
   switch(m_nEncoderType)
      {
      case -1:
         encoder = NULL;
         break;
      case 0:
         encoder = new libcomm::nrcc(m_mbGenerator);
         break;
      case 1:
         encoder = new libcomm::rscc(m_mbGenerator);
         break;
      }
   // create codec
   libcomm::codec *cdc;
   switch(m_nCodecType)
      {
      case -1:
         cdc = NULL;
         break;
      case 0:
         cdc = new libcomm::uncoded(*encoder, m_nTau);
         break;
      case 1:
         switch(m_nArithmetic)
            {
            case 0:
               cdc = new libcomm::mapcc<libbase::logreal>(*encoder, m_nTau, m_bTerminated);
               break;
            case 1:
               cdc = new libcomm::mapcc<libbase::logrealfast>(*encoder, m_nTau, m_bTerminated);
               break;
            case 2:
               cdc = new libcomm::mapcc<libbase::mpreal>(*encoder, m_nTau, m_bTerminated);
               break;
            case 3:
               cdc = new libcomm::mapcc<libbase::mpgnu>(*encoder, m_nTau, m_bTerminated);
               break;
            }
         break;
      case 2:
         switch(m_nArithmetic)
            {
            case 0:
               cdc = new libcomm::turbo<libbase::logreal>(*encoder, m_nTau, m_vpInterleavers, m_nIterations, m_bSimile, m_bTerminated, m_bParallel);
               break;
            case 1:
               cdc = new libcomm::turbo<libbase::logrealfast>(*encoder, m_nTau, m_vpInterleavers, m_nIterations, m_bSimile, m_bTerminated, m_bParallel);
               break;
            case 2:
               cdc = new libcomm::turbo<libbase::mpreal>(*encoder, m_nTau, m_vpInterleavers, m_nIterations, m_bSimile, m_bTerminated, m_bParallel);
               break;
            case 3:
               cdc = new libcomm::turbo<libbase::mpgnu>(*encoder, m_nTau, m_vpInterleavers, m_nIterations, m_bSimile, m_bTerminated, m_bParallel);
               break;
            }
         break;
      case 3:
         break;
      }
   // clean up and return
   if(encoder != NULL)
      delete encoder;
   return cdc;
   }
