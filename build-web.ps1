# Builds the browser demo into ./web/pkg.
# Requirements: rustup target add wasm32-unknown-unknown
#               cargo install wasm-bindgen-cli --version <same as Cargo.lock>
# Serve:        python -m http.server 8080 --directory web   ->  http://localhost:8080
param([switch]$Debug)

$ErrorActionPreference = "Stop"
$profileDir = if ($Debug) { "debug" } else { "release" }
$cargoArgs = @("build", "-p", "helios-demo", "--lib", "--target", "wasm32-unknown-unknown")
if (-not $Debug) { $cargoArgs += "--release" }

cargo @cargoArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

wasm-bindgen --target web --no-typescript --out-dir web/pkg "target/wasm32-unknown-unknown/$profileDir/helios_demo.wasm"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$size = (Get-Item web/pkg/helios_demo_bg.wasm).Length / 1MB
Write-Host ("Built web/pkg ({0:N1} MiB wasm)" -f $size)
