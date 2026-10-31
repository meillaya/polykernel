# Excalidraw CLI flake

Package `@excalidraw/cli` 0.2.0 for Nix. The CLI renders local `.excalidraw`
scenes to PNG and needs a Chrome, Chromium, Edge, or Firefox binary at runtime.

The flake supports aarch64-darwin and Linux. nixpkgs 26.11 dropped
x86_64-darwin; pin nixpkgs-26.05 if you need an Intel Mac.

## Run

```bash
nix run .#excalidraw-cli -- --version

nix run .#excalidraw-cli -- render scene.excalidraw \
  --out scene.png --scale 2 \
  --browser-path /path/to/chrome
```

On Linux, the browser variant supplies Chromium:

```bash
nix run .#excalidraw-cli-browser -- render scene.excalidraw --out scene.png --scale 2
```

On macOS, point `EXCALIDRAW_BROWSER_PATH` at a Chrome or Chromium binary, or pass
`--browser-path`.

## Build and develop

```bash
nix build .#excalidraw-cli
./result/bin/excalidraw --version

nix develop
```

The development shell contains the CLI and Node 22. On Linux it also contains
Chromium and exports `EXCALIDRAW_BROWSER_PATH`.

## Update

1. Change the version in `npm/package.json`.
2. Regenerate `npm/package-lock.json` with
   `npm install --package-lock-only --ignore-scripts`.
3. Refresh the dependency hash with
   `nix run nixpkgs#prefetch-npm-deps -- npm/package-lock.json` and set
   `npmDepsHash` in `package.nix`.
4. Run `nix build .#excalidraw-cli` and render a scene.
