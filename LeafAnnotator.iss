[Setup]
AppName=植物表型标注工具
AppVersion=0.2.0
DefaultDirName={autopf}\植物表型标注工具
DefaultGroupName=植物表型标注工具
OutputDir=dist_installer
OutputBaseFilename=LeafAnnotator_Setup
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64

[Files]
Source: "dist\LeafAnnotator\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\植物表型标注工具"; Filename: "{app}\LeafAnnotator.exe"
Name: "{commondesktop}\植物表型标注工具"; Filename: "{app}\LeafAnnotator.exe"
