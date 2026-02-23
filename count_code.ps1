param(
    [string]$Path = ".",
    [switch]$IncludeBlank,
    [switch]$Verbose
)

$excludeDirs = @(
    "node_modules",
    "bin",
    "obj",
    ".angular",
    ".vs",
    "__pycache__",
    ".git",
    "dist",
    "build",
    "coverage",
    ".venv",
    "venv",
    "packages"
)

$languages = @{
    "C#"         = @("*.cs")
    "TypeScript" = @("*.ts", "*.tsx")
    "JavaScript" = @("*.js", "*.jsx", "*.mjs")
    "Python"     = @("*.py")
    "HTML"       = @("*.html", "*.htm")
    "CSS"        = @("*.css")
    "SCSS/SASS"  = @("*.scss", "*.sass")
    "JSON"       = @("*.json")
    "YAML"       = @("*.yaml", "*.yml")
    "XML"        = @("*.xml", "*.csproj", "*.sln", "*.config")
    "Markdown"   = @("*.md")
    "Shell"      = @("*.sh", "*.bash")
    "PowerShell" = @("*.ps1", "*.psm1")
    "Batch"      = @("*.bat", "*.cmd")
    "SQL"        = @("*.sql")
    "Docker"     = @("Dockerfile", "*.dockerfile", "docker-compose*.yml")
}

function Get-ExcludePattern {
    $patterns = @()
    foreach ($dir in $excludeDirs) {
        $patterns += "*\$dir\*"
        $patterns += "*/$dir/*"
    }
    return $patterns
}

function Test-ExcludedPath {
    param([string]$FilePath)

    foreach ($dir in $excludeDirs) {
        if ($FilePath -match "[\\/]$dir[\\/]") {
            return $true
        }
    }
    return $false
}

function Count-LinesInFile {
    param([string]$FilePath)

    try {
        $content = Get-Content -Path $FilePath -ErrorAction Stop
        if ($IncludeBlank) {
            return $content.Count
        } else {
            return ($content | Where-Object { $_.Trim() -ne "" }).Count
        }
    } catch {
        return 0
    }
}

$results = @{}
$fileCountByLang = @{}
$totalLines = 0
$totalFiles = 0

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "   LINES OF CODE COUNTER" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan
Write-Host "Scanning directory: $((Resolve-Path $Path).Path)" -ForegroundColor Yellow
Write-Host "Excluding: $($excludeDirs -join ', ')`n" -ForegroundColor DarkGray

foreach ($lang in $languages.Keys) {
    $results[$lang] = 0
    $fileCountByLang[$lang] = 0

    foreach ($pattern in $languages[$lang]) {
        $files = Get-ChildItem -Path $Path -Filter $pattern -Recurse -File -ErrorAction SilentlyContinue

        foreach ($file in $files) {
            if (-not (Test-ExcludedPath -FilePath $file.FullName)) {
                $lines = Count-LinesInFile -FilePath $file.FullName
                $results[$lang] += $lines
                $fileCountByLang[$lang]++
                $totalLines += $lines
                $totalFiles++

                if ($Verbose) {
                    Write-Host "  $($file.FullName): $lines lines" -ForegroundColor DarkGray
                }
            }
        }
    }
}

# Sort results by line count (descending)
$sortedResults = $results.GetEnumerator() | Where-Object { $_.Value -gt 0 } | Sort-Object -Property Value -Descending

Write-Host "`n----------------------------------------" -ForegroundColor Gray
Write-Host " RESULTS BY LANGUAGE" -ForegroundColor Green
Write-Host "----------------------------------------" -ForegroundColor Gray
Write-Host ("{0,-15} {1,10} {2,10} {3,10}" -f "Language", "Lines", "Files", "% of Total") -ForegroundColor White

foreach ($item in $sortedResults) {
    $lang = $item.Key
    $lines = $item.Value
    $files = $fileCountByLang[$lang]
    $percentage = if ($totalLines -gt 0) { [math]::Round(($lines / $totalLines) * 100, 1) } else { 0 }

    $color = switch ($lang) {
        "C#"         { "Magenta" }
        "TypeScript" { "Blue" }
        "JavaScript" { "Yellow" }
        "Python"     { "Green" }
        "HTML"       { "Red" }
        "CSS"        { "Cyan" }
        "SCSS/SASS"  { "Cyan" }
        default      { "White" }
    }

    Write-Host ("{0,-15} {1,10:N0} {2,10} {3,9}%" -f $lang, $lines, $files, $percentage) -ForegroundColor $color
}

Write-Host "----------------------------------------" -ForegroundColor Gray
Write-Host ("{0,-15} {1,10:N0} {2,10}" -f "TOTAL", $totalLines, $totalFiles) -ForegroundColor White
Write-Host "========================================`n" -ForegroundColor Cyan

# Return results as object for programmatic use
return [PSCustomObject]@{
    Languages = $results
    TotalLines = $totalLines
    TotalFiles = $totalFiles
}