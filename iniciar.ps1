# Iniciar LocalTranscription
Clear-Host
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "   Iniciando LocalTranscription v2.0     " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

$pythonPath = "$env:USERPROFILE\.conda\envs\localtranscription\python.exe"

if (Test-Path $pythonPath) {
    Write-Host "[+] Entorno conda 'localtranscription' detectado." -ForegroundColor Green
    Write-Host "[+] Iniciando servidor Flask y WebSockets en http://127.0.0.1:5000..." -ForegroundColor Green
    Write-Host "[+] Abriendo navegador por defecto..." -ForegroundColor Yellow
    Write-Host ""
    
    # Wait 1.5 seconds and launch default browser
    Start-Sleep -Seconds 1
    Start-Process "http://127.0.0.1:5000"
    
    # Run the Flask application using the env python binary directly
    & $pythonPath app.py
} else {
    Write-Host "[-] ERROR: No se encontró el entorno 'localtranscription' en:" -ForegroundColor Red
    Write-Host "    $pythonPath" -ForegroundColor Red
    Write-Host ""
    Write-Host "[i] Por favor asegúrate de haber creado el entorno mediante conda:" -ForegroundColor Yellow
    Write-Host "    conda env create -f environment.yml" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Presiona Enter para cerrar..."
}
