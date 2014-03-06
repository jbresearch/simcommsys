Imports System
Imports System.Collections.Generic
Imports Extensibility
Imports EnvDTE
Imports EnvDTE80
Imports Microsoft.VisualStudio.VCProjectEngine
Imports VSLangProj

Public Class Connect
    Implements IDTExtensibility2

    Private DTE As EnvDTE80.DTE2
    Private WithEvents BuildEvents As EnvDTE.BuildEvents

    Public Sub OnConnection(ByVal application As Object, ByVal connectMode As ext_ConnectMode, _
       ByVal addInInst As Object, ByRef custom As Array) Implements IDTExtensibility2.OnConnection

        Me.DTE = CType(application, EnvDTE80.DTE2)

        Select Case connectMode

            Case ext_ConnectMode.ext_cm_AfterStartup

                ' The add-in has been loaded after Visual Studio was loaded. Since Visual Studio
                ' is fully initialized, we can initialize the add-in
                InitializeAddIn()

            Case ext_ConnectMode.ext_cm_Startup

                ' The add-in has been loaded with Visual Studio. Do nothing until Visual Studio 
                ' is fully initialized (the OnStartupComplete method will be called)

            Case ext_ConnectMode.ext_cm_UISetup

                ' Do nothing in this case

        End Select

    End Sub

    Public Sub OnStartupComplete(ByRef custom As Array) Implements IDTExtensibility2.OnStartupComplete

        InitializeAddIn()

    End Sub

    Private Sub InitializeAddIn()

        Me.BuildEvents = Me.DTE.Events.BuildEvents

    End Sub

    Public Sub OnDisconnection(ByVal disconnectMode As ext_DisconnectMode, ByRef custom As Array) _
       Implements IDTExtensibility2.OnDisconnection

        Me.BuildEvents = Nothing

    End Sub

    Public Sub OnAddInsUpdate(ByRef custom As Array) Implements IDTExtensibility2.OnAddInsUpdate
    End Sub

    Public Sub OnBeginShutdown(ByRef custom As Array) Implements IDTExtensibility2.OnBeginShutdown
    End Sub

    Sub WriteToOutputBuildPane(ByVal Message As String)
        DTE.Windows.Item(EnvDTE.Constants.vsWindowKindOutput).Object.OutputWindowPanes.Item("Build").OutputString(Message & vbCrLf)
    End Sub


    Private Sub BuildEvents_OnBuildBegin(ByVal Scope As EnvDTE.vsBuildScope, ByVal Action As EnvDTE.vsBuildAction) Handles BuildEvents.OnBuildBegin

        If GetName(DTE.Solution.FullName.ToLower()) = "simcommsys.sln" Then
            Dim project As EnvDTE.Project
            Dim vcconfigs As IVCCollection
            Dim branch As String
            Dim liblocation As String
            GetHandles(project, vcconfigs)

            branch = GetBranch(DTE.Solution.FullName)
            WriteToOutputBuildPane("Building solution in the following branch: " & branch)

            'Get the location of the boost libraries depending on the chosen platform 32-bit or 64-bit and MSVC version
            liblocation = GetLibLocation(project)
            WriteToOutputBuildPane("Using boost library folder: " + liblocation)
            SetEnvironment(vcconfigs, branch, liblocation)
        End If

    End Sub

    Private Sub GetHandles(ByRef project As EnvDTE.Project, ByRef vcconfigs As IVCCollection)

        Dim projectName As String

        'Use SimCommsys to get access to the property sheet common to all projects
        projectName = "SimCommsys"
        Dim projectitem As EnvDTE.ProjectItem = GetProject(projectName)
        project = projectitem.Object()
        Dim vsPrj As VSProject = CType(project.Object, VSProject)
        Dim vcproj As VCProject = vsPrj.Project.Object
        vcconfigs = vcproj.Configurations

    End Sub

    Private Sub SetEnvironment(ByVal vcconfigs As IVCCollection, ByVal branch As String, ByVal liblocation As String)

        Dim PropertyFound As Boolean
        PropertyFound = False
        For Each vcconfig As VCConfiguration In vcconfigs
            'Find the required property sheet
            For Each prop In vcconfig.PropertySheets
                If prop.Name.ToLower() = "simcommsyspropsheet" Then
                    SetUserMacro("Branch", branch, prop)
                    SetUserMacro("BOOSTLIB", liblocation, prop)
                    PropertyFound = True
                    Exit For
                End If
            Next
            If PropertyFound = True Then
                Exit For
            End If
        Next
    End Sub

    Sub SetUserMacro(ByVal MacroName, ByVal Value, ByVal prop)
        Dim PropertyFound As Boolean
        Dim BranchFound As Boolean
        Dim um As VCUserMacro
        Dim um2 As VCUserMacro

        PropertyFound = True
        For Each um In prop.UserMacros
            If um.Name = MacroName Then
                'If the MacroName value was already set properly then do not save this
                'This should avoid some rebuilds
                If um.value <> Value Then
                    um.Value = Value
                    um.PerformEnvironmentSet = True
                    prop.Save()
                End If
                BranchFound = True
                Exit For
            End If
        Next
        'If the Macro "Branch" was not defined, then add it now
        If BranchFound = False Then
            um2 = prop.AddUserMacro(MacroName, Value)
            um2.PerformEnvironmentSet = True
            prop.Save()
        End If
    End Sub

    Function GetProject(ByVal projectName As String) As EnvDTE.ProjectItem
        For Each project In DTE.Solution.Projects
            If (project.ConfigurationManager IsNot Nothing) Then
                ' It's a project!
                If (project.Name.ToLower() = projectName.ToLower()) Then Return project
            Else
                If (project.ProjectItems IsNot Nothing) Then
                    For Each projectItem In project.ProjectItems
                        If (projectItem.SubProject IsNot Nothing) Then
                            If projectItem.Name.ToLower() = projectName.ToLower() Then Return projectItem
                        End If
                    Next
                End If
            End If
        Next
    End Function


    Function GetName(ByVal fullname As String) As String
        Dim length As Integer
        Dim ch As String

        length = fullname.Length
        GetName = ""
        For i = length To 1 Step -1
            ch = Mid(fullname, i, 1)
            If ch <> "\" Then
                GetName = ch + GetName
            Else
                i = 0
            End If
        Next
    End Function

    Function GetBranch(ByVal solution As String) As String

#If GIT Then

        Dim tmp As String
        Dim folder As String
        Dim cmd As String
        Dim slash As String

        'Get the branch name
        tmp = IO.Path.GetTempFileName()
        folder = IO.Path.GetDirectoryName(solution)
        cmd = "cmd /c ""cd /d " + folder + " & git rev-parse --abbrev-ref HEAD >" + tmp + """"
        Shell(cmd, AppWinStyle.Hide, True)
        GetBranch = My.Computer.FileSystem.ReadAllText(tmp)
        'Remove the final carriage return, if present
        If Right(GetBranch, 1) = vbLf Then GetBranch = Left(GetBranch, Len(GetBranch) - 1)
        'Pick the last name if the branch contains forward slashes
        slash = InStrRev(GetBranch, "/")
        If slash > -1 Then GetBranch = Right(GetBranch, GetBranch.Length - slash)
        My.Computer.FileSystem.DeleteFile(tmp)

#Else

        'Get the branch name
        Dim ptrstart, ptrend As Integer
        ptrstart = 0
        ptrend = 0
        For i = Len(DTE.Solution.FullName) To 1 Step -1
            If Mid(DTE.Solution.FullName, i, 1) = "\" Then
                If ptrend = 0 Then
                    ptrend = i
                Else
                    ptrstart = i
                    i = 0
                End If
            End If
        Next

        GetBranch = Mid(DTE.Solution.FullName, ptrstart + 1, ptrend - ptrstart - 1)


#End If


    End Function

    Private Sub BuildEvents_OnBuildDone(ByVal Scope As EnvDTE.vsBuildScope, ByVal Action As EnvDTE.vsBuildAction) Handles BuildEvents.OnBuildDone
        If GetName(DTE.Solution.FullName.ToLower()) = "simcommsys.sln" Then
            Dim project As EnvDTE.Project
            Dim vcconfigs As IVCCollection
            GetHandles(project, vcconfigs)
            SetEnvironment(vcconfigs, "Value_is_set_automatically_by_macro", "Value_is_set_automatically_by_macro")
        End If
    End Sub

    Private Sub BuildEvents_OnBuildProjConfigDone(ByVal Project As String, ByVal ProjectConfig As String, ByVal Platform As String, ByVal SolutionConfig As String, ByVal Success As Boolean) Handles BuildEvents.OnBuildProjConfigDone
        If Success = False Then 'The build failed...cancel any further builds.    
            DTE.ExecuteCommand("Build.Cancel")
        End If
    End Sub

    Private Function GetLibLocation(ByVal project As EnvDTE.Project) As String

        Dim pf As String = ""
        If project.ConfigurationManager.ActiveConfiguration.PlatformName.ToLower() = "win32" Then pf = 32
        If project.ConfigurationManager.ActiveConfiguration.PlatformName.ToLower() = "x64" Then pf = 64
        GetLibLocation = "lib" + pf + "-msvc-" + DTE.Version
    End Function


End Class