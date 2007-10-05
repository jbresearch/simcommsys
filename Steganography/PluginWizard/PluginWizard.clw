; CLW file contains information for the MFC ClassWizard

[General Info]
Version=1
LastClass=CPluginWizardDlg
LastTemplate=CDialog
NewFileInclude1=#include "stdafx.h"
NewFileInclude2=#include "PluginWizard.h"

ClassCount=3
Class1=CPluginWizardApp
Class2=CPluginWizardDlg
Class3=CAboutDlg

ResourceCount=3
Resource1=IDD_ABOUTBOX
Resource2=IDR_MAINFRAME
Resource3=IDD_PLUGINWIZARD_DIALOG

[CLS:CPluginWizardApp]
Type=0
HeaderFile=PluginWizard.h
ImplementationFile=PluginWizard.cpp
Filter=N

[CLS:CPluginWizardDlg]
Type=0
HeaderFile=PluginWizardDlg.h
ImplementationFile=PluginWizardDlg.cpp
Filter=D
BaseClass=CDialog
VirtualFilter=dWC
LastObject=IDC_TYPE

[CLS:CAboutDlg]
Type=0
HeaderFile=PluginWizardDlg.h
ImplementationFile=PluginWizardDlg.cpp
Filter=D
LastObject=CAboutDlg

[DLG:IDD_ABOUTBOX]
Type=1
Class=CAboutDlg
ControlCount=4
Control1=IDC_STATIC,static,1342177283
Control2=IDC_STATIC,static,1342308480
Control3=IDC_STATIC,static,1342308352
Control4=IDOK,button,1342373889

[DLG:IDD_PLUGINWIZARD_DIALOG]
Type=1
Class=CPluginWizardDlg
ControlCount=8
Control1=IDOK,button,1342242817
Control2=IDCANCEL,button,1342242816
Control3=IDC_TYPE,combobox,1344339971
Control4=IDC_STATIC,static,1342308352
Control5=IDC_OLDNAME,edit,1350631552
Control6=IDC_STATIC,static,1342308352
Control7=IDC_NEWNAME,edit,1350631552
Control8=IDC_STATIC,static,1342308352

