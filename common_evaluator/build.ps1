param(
    [string]$Python = 'python',
    [string]$Output = (Join-Path $PSScriptRoot 'dist')
)
$ErrorActionPreference = 'Stop'
$source = $PSScriptRoot
$relativeNames = @('assets.py','metrics.py','report.py','engine.py','main.py','gui.py','protocol.json')
$outputPath = [IO.Path]::GetFullPath($Output)
if (Test-Path -LiteralPath $outputPath) { throw "Choose a new output path: $outputPath" }
$work = Join-Path $outputPath 'build'
New-Item -ItemType Directory -Path $work | Out-Null
$packageData = & $Python -c "import platform, tkinter; import importlib.metadata as m; import json; tk = tkinter.Tcl().eval('info patchlevel'); print(json.dumps(dict(python=platform.python_version(),platform=platform.platform(),tk=tk,packages={n:m.version(n) for n in ('numpy','Pillow','pyinstaller','pyinstaller-hooks-contrib','altgraph','packaging','pefile','pywin32-ctypes','setuptools')})))"
if ($LASTEXITCODE -ne 0) { throw 'Python, Tk or build dependencies are missing' }
$info = @{ version = '0.1.0'; build_environment = ($packageData | ConvertFrom-Json); source_sha256 = @{} }
foreach ($name in $relativeNames) {
    $info.source_sha256[$name] = (Get-FileHash -LiteralPath (Join-Path $source $name)).Hash.ToLowerInvariant()
}
$metadata = Join-Path $work 'build_info.json'
[IO.File]::WriteAllText($metadata, (ConvertTo-Json $info -Depth 6), [Text.UTF8Encoding]::new($false))
$opts = @('--onefile','--noconfirm','--distpath',$outputPath,'--workpath',$work,'--specpath',$work,
          '--paths',$source,'--add-data',((Join-Path $source 'protocol.json')+';.'),'--add-data',($metadata+';.'))
& $Python -m PyInstaller @opts --windowed --name 'UC-FoodWaste-Evaluator' (Join-Path $source 'gui.py')
if ($LASTEXITCODE -ne 0) { throw 'GUI build failed' }
& $Python -m PyInstaller @opts --console --name 'ucfw-evaluate' (Join-Path $source 'main.py')
if ($LASTEXITCODE -ne 0) { throw 'CLI build failed' }
Copy-Item -LiteralPath $metadata -Destination (Join-Path $outputPath 'build_info.json')
$hashes = Get-ChildItem -LiteralPath $outputPath -Filter '*.exe' | Get-FileHash -Algorithm SHA256
$hashes | ForEach-Object { "$($_.Hash.ToLowerInvariant())  $(Split-Path -Leaf $_.Path)" } |
    Set-Content -LiteralPath (Join-Path $outputPath 'SHA256SUMS.txt') -Encoding ascii
$hashes
