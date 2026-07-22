{
  description = "PolyKernel - out-of-tree C++/MLIR compiler + GPU kernel toolchain (dev shell)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    # devenv 2.x flakes integration (devenv.lib.mkShell). Pin to v2.1.2 (the
    # currently installed CLI) so it is current with nixpkgs-unstable.
    devenv.url = "github:cachix/devenv/v2.1.2";
    devenv.inputs.nixpkgs.follows = "nixpkgs";
  };

  nixConfig = {
    extra-trusted-public-keys = "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw=";
    extra-substituters = "https://devenv.cachix.org";
  };

  # BUILD MODEL (Pinned contract A): the deliverable is a devenv devShell +
  # manual cmake inside the shell - NOT a sandboxed `nix build` of polykernel.
  # This flake exposes devShells.${system}.default (via devenv), not a package.
  outputs = { self, nixpkgs, devenv, ... } @ inputs:
    let
      system = "x86_64-linux";
      # nixpkgs config (allowUnfree for the CUDA EULA, cudaSupport/rocmSupport
      # for the GPU redistributables) is applied HERE, when importing nixpkgs.
      # devenv.lib.mkShell uses the pkgs handed to it and only overlays the
      # module's `overlays`; it does not re-read a `nixpkgs.config` module
      # option, so the config must be set on the nixpkgs instantiation.
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
          rocmSupport = true;
          # sm_80 (A100) / sm_90 (H100) for the rental-validated CUDA path.
          cudaCapabilities = [ "8.0" "8.6" "8.9" "9.0" ];
        };
      };
    in
    {
      devShells.${system}.default = devenv.lib.mkShell {
        inherit inputs pkgs;
        modules = [ ./devenv.nix ];
      };
    };
}
