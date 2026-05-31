# Iniciar LocalTranscription
Clear-Host
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "   Iniciando LocalTranscription v2.0     " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

$envRoot    = "$env:USERPROFILE\.conda\envs\localtranscription"
$pythonPath = "$envRoot\python.exe"

if (-not (Test-Path $pythonPath)) {
    Write-Host "[-] ERROR: No se encontro el entorno 'localtranscription' en:" -ForegroundColor Red
    Write-Host "    $pythonPath" -ForegroundColor Red
    Write-Host ""
    Write-Host "[i] Crealo con: conda env create -f environment.yml" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Presiona Enter para cerrar..."
    exit 1
}

Write-Host "[+] Entorno conda 'localtranscription' detectado." -ForegroundColor Green
Write-Host "[+] Iniciando servidor Flask + WebSockets..." -ForegroundColor Yellow
Write-Host "[i] El modelo ASR pesa ~2.4GB y tarda ~60s en cargar en CPU. Espera." -ForegroundColor DarkGray
Write-Host ""

# Make ffmpeg & co reachable
$env:PATH = "$envRoot;$envRoot\Library\bin;$envRoot\Scripts;$env:PATH"

# Launch python in background; its stdout/stderr goes to this console
$proc = Start-Process -FilePath $pythonPath -ArgumentList "app.py" -NoNewWindow -PassThru

# Poll the server until model_loaded:true. Print a tick every ~10s.
$url       = "http://127.0.0.1:5000/api/settings"
$start     = Get-Date
$lastTick  = $start
$ready     = $false

while (-not $proc.HasExited) {
    Start-Sleep -Seconds 2
    try {
        $r = Invoke-RestMethod -Uri $url -TimeoutSec 2 -ErrorAction Stop
        if ($r.model_loaded) { $ready = $true; break }
    } catch {
        # Server not up yet or still loading -- keep waiting silently
    }
    if (((Get-Date) - $lastTick).TotalSeconds -ge 10) {
        $elapsed = [int]((Get-Date) - $start).TotalSeconds
        Write-Host "[.] Aun cargando modelo ASR... ${elapsed}s" -ForegroundColor DarkGray
        $lastTick = Get-Date
    }
}

if ($ready) {
    $elapsed = [int]((Get-Date) - $start).TotalSeconds
    Write-Host ""
    Write-Host "=========================================" -ForegroundColor Green
    Write-Host " [OK] Servidor listo en ${elapsed}s"        -ForegroundColor Green
    Write-Host " URL: http://127.0.0.1:5000"                -ForegroundColor Green
    Write-Host "=========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "[+] Abriendo navegador por defecto..." -ForegroundColor Green
    Start-Process "http://127.0.0.1:5000"
    Write-Host ""
    Write-Host "[i] Ctrl+C para detener el servidor." -ForegroundColor DarkGray

    Wait-Process -Id $proc.Id
} else {
    Write-Host ""
    Write-Host "[-] El proceso de Python termino antes de cargar el modelo." -ForegroundColor Red
    Write-Host "[i] Revisa logs/localtranscription.log para mas detalles." -ForegroundColor Yellow
    Read-Host "Presiona Enter para cerrar..."
}
