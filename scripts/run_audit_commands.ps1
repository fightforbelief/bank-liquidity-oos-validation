
Set-Location "C:\Users\hanqi\Desktop\projects\p12-bank-liquidity-oos-validation"
$env:PYTHONPATH = "src"
$py = ".\.venv\Scripts\python.exe"
& $py -m p12_bank_liquidity.cli env-audit
Write-Output ("env_exit=" + $LASTEXITCODE)
& $py -m p12_bank_liquidity.cli source-audit
Write-Output ("source_exit=" + $LASTEXITCODE)
& $py -m p12_bank_liquidity.cli target-audit
Write-Output ("target_exit=" + $LASTEXITCODE)
& $py -m p12_bank_liquidity.cli vintage-audit
Write-Output ("vintage_exit=" + $LASTEXITCODE)
& $py -m p12_bank_liquidity.cli report
Write-Output ("report_exit=" + $LASTEXITCODE)
& $py -m p12_bank_liquidity.cli data-audit
Write-Output ("data_audit_exit=" + $LASTEXITCODE)
& $py -m p12_bank_liquidity.cli reproduce
Write-Output ("reproduce_exit=" + $LASTEXITCODE)
& $py -m p12_bank_liquidity.cli validate-later-period
Write-Output ("later_exit=" + $LASTEXITCODE)
