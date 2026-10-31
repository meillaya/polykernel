{
  description = "Excalidraw CLI, packaged for Nix";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs =
    { self, nixpkgs }:
    let
      # nixpkgs 26.11 dropped x86_64-darwin; pin nixpkgs-26.05 if you need it.
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
      forAllSystems =
        f: nixpkgs.lib.genAttrs systems (system: f (import nixpkgs { inherit system; }));
    in
    {
      packages = forAllSystems (
        pkgs:
        let
          cli = pkgs.callPackage ./package.nix { };
        in
        {
          excalidraw-cli = cli;
          default = cli;
        }
        // nixpkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
          # Chromium is the browser the CLI drives on Linux. On Darwin, use a
          # host Chrome or Chromium through EXCALIDRAW_BROWSER_PATH.
          excalidraw-cli-browser = pkgs.callPackage ./package.nix {
            browser = nixpkgs.lib.getExe pkgs.chromium;
          };
        }
      );

      apps = forAllSystems (
        pkgs:
        let
          system = pkgs.stdenv.hostPlatform.system;
          cli = self.packages.${system}.excalidraw-cli;
        in
        {
          default = {
            type = "app";
            program = nixpkgs.lib.getExe cli;
          };
          excalidraw-cli = {
            type = "app";
            program = nixpkgs.lib.getExe cli;
          };
        }
        // nixpkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
          excalidraw-cli-browser = {
            type = "app";
            program = nixpkgs.lib.getExe self.packages.${pkgs.stdenv.hostPlatform.system}.excalidraw-cli-browser;
          };
        }
      );

      devShells = forAllSystems (
        pkgs:
        let
          isLinux = pkgs.stdenv.hostPlatform.isLinux;
        in
        {
          default = pkgs.mkShell {
            packages = [
              self.packages.${pkgs.stdenv.hostPlatform.system}.excalidraw-cli
              pkgs.nodejs_22
            ]
            ++ nixpkgs.lib.optional isLinux pkgs.chromium;

            shellHook = nixpkgs.lib.optionalString isLinux ''
              export EXCALIDRAW_BROWSER_PATH="${nixpkgs.lib.getExe pkgs.chromium}"
            '' + nixpkgs.lib.optionalString (!isLinux) ''
              echo "Excalidraw CLI ready. Set EXCALIDRAW_BROWSER_PATH to a Chrome or Chromium binary before rendering."
            '';
          };
        }
      );

      checks = forAllSystems (
        pkgs:
        {
          excalidraw-cli-version = pkgs.runCommand "excalidraw-cli-version" {
            nativeBuildInputs = [ self.packages.${pkgs.stdenv.hostPlatform.system}.excalidraw-cli ];
          } ''
            excalidraw --version
            touch $out
          '';
        }
      );
    };
}
