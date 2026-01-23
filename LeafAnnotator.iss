[Setup]
AppName=植物表型标注工具
AppVersion=0.1.0
DefaultDirName={autopf}\LeafAnnotator
DefaultGroupName=LeafAnnotator
OutputDir=dist_installer
OutputBaseFilename=LeafAnnotator_Setup
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64

[Files]
Source: "dist\LeafAnnotator\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\LeafAnnotator"; Filename: "{app}\LeafAnnotator.exe"
Name: "{commondesktop}\LeafAnnotator"; Filename: "{app}\LeafAnnotator.exe"
