; CLW file contains information for the MFC ClassWizard

[General Info]
Version=1
LastClass=CFilterEmbedDlg
LastTemplate=CDialog
NewFileInclude1=#include "stdafx.h"
NewFileInclude2=#include "filterembed.h"
LastPage=0

ClassCount=4
Class1=CFilterEmbedApp
Class2=CAboutDlg
Class3=CFilterEmbedDlg

ResourceCount=3
Resource1=IDD_DIALOG1
Resource2=IDD_ABOUTBOX
Class4=CComputeStrengthDlg
Resource3=IDD_COMPUTE_STRENGTH

[CLS:CFilterEmbedApp]
Type=0
BaseClass=CWinApp
HeaderFile=FilterEmbed.h
ImplementationFile=FilterEmbed.cpp

[CLS:CAboutDlg]
Type=0
BaseClass=CDialog
HeaderFile=FilterEmbedDlg.h
ImplementationFile=FilterEmbedDlg.cpp

[CLS:CFilterEmbedDlg]
Type=0
BaseClass=CDialog
HeaderFile=FilterEmbedDlg.h
ImplementationFile=FilterEmbedDlg.cpp
Filter=D
VirtualFilter=dWC
LastObject=CFilterEmbedDlg

[DLG:IDD_ABOUTBOX]
Type=1
Class=CAboutDlg
ControlCount=3
Control1=65535,static,1342308480
Control2=65535,static,1342308352
Control3=IDOK,button,1342373889

[DLG:IDD_DIALOG1]
Type=1
Class=CFilterEmbedDlg
ControlCount=47
Control1=IDOK,button,1342242817
Control2=IDCANCEL,button,1342242816
Control3=IDC_EMBED_STRENGTH,edit,1350631552
Control4=IDC_STATIC,static,1342308352
Control5=IDC_STATIC,static,1342308352
Control6=IDC_SOURCE,edit,1350633600
Control7=IDC_LOAD_SOURCE,button,1342242816
Control8=IDC_STATIC,static,1342308352
Control9=IDC_INFO_FILESIZE,edit,1350641792
Control10=IDC_STATIC,static,1342308352
Control11=IDC_INFO_USAGE,edit,1350641792
Control12=IDC_STATIC,static,1342308352
Control13=IDC_INFO_CAPACITY,edit,1350641792
Control14=IDC_INTERLEAVER_DENSITY,edit,1350631552
Control15=IDC_STATIC,static,1342308352
Control16=IDC_STATIC,button,1342177287
Control17=IDC_STATIC,static,1342308352
Control18=IDC_CODEC,edit,1350633600
Control19=IDC_STATIC,static,1342308352
Control20=IDC_PUNCTURE,edit,1350633600
Control21=IDC_LOAD_CODEC,button,1342242816
Control22=IDC_LOAD_PUNCTURE,button,1342242816
Control23=IDC_CLEAR_SOURCE,button,1342242816
Control24=IDC_CLEAR_CODEC,button,1342242816
Control25=IDC_CLEAR_PUNCTURE,button,1342242816
Control26=IDC_INTERLEAVE,button,1342242819
Control27=IDC_COMPUTE_STRENGTH,button,1342242816
Control28=IDC_STATIC,static,1342308352
Control29=IDC_INFO_USABLE,edit,1350641792
Control30=IDC_STATIC,static,1342308352
Control31=IDC_INFO_CODESIZE,edit,1350641792
Control32=IDC_STATIC,static,1342308352
Control33=IDC_INFO_DATARATE,edit,1350641792
Control34=IDC_STATIC,static,1342308352
Control35=IDC_SOURCE_TYPE,combobox,1344339971
Control36=IDC_INTERLEAVER_SEED,edit,1350631552
Control37=IDC_STATIC,static,1342308352
Control38=IDC_SOURCE_SEED,edit,1350631552
Control39=IDC_STATIC,static,1342308352
Control40=IDC_STATIC,button,1342177287
Control41=IDC_STATIC,button,1342177287
Control42=IDC_STATIC,button,1342177287
Control43=IDC_STATIC,button,1342177287
Control44=IDC_EMBED_SEED,edit,1350631552
Control45=IDC_STATIC,static,1342308352
Control46=IDC_STATIC,static,1342308352
Control47=IDC_EMBED_RATE,edit,1350631552

[DLG:IDD_COMPUTE_STRENGTH]
Type=1
Class=CComputeStrengthDlg
ControlCount=4
Control1=IDOK,button,1342242817
Control2=IDCANCEL,button,1342242816
Control3=IDC_POWER,edit,1350631552
Control4=IDC_STATIC,static,1342308352

[CLS:CComputeStrengthDlg]
Type=0
HeaderFile=ComputeStrengthDlg.h
ImplementationFile=ComputeStrengthDlg.cpp
BaseClass=CDialog
Filter=D
VirtualFilter=dWC

