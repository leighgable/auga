{
  description = "torch cpu using uv2nix";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      ...
    }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      overlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
      };

      editableOverlay = workspace.mkEditablePyprojectOverlay {
        root = "$REPO_ROOT";
      };

      pythonSets = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
            };
          };
          python = pkgs.python314;

          hacks = pkgs.callPackage pyproject-nix.build.hacks { };

          customOverlay =
            final: prev:
            let
              ale-py-version = "0.12.0";

              ale-py-roms = pkgs.stdenvNoCC.mkDerivation {
                pname = "ale-py-roms";
                version = ale-py-version;
                src = pkgs.fetchurl {
                  url = "https://files.pythonhosted.org/packages/10/c3/2231ceb5bfba7056818cb1f460383c14b8a417939c978eb44c3b55514f14/ale_py-${ale-py-version}-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
                  sha256 = "1m7bxhjwc51l6wy9wwpa386cz635bh08b2rnlbg6nlqgb17yy1h0";
                };
                nativeBuildInputs = [ pkgs.unzip ];
                dontUnpack = true;
                installPhase = ''
                  unzip $src
                  mkdir -p $out
                  cp -r ale_py/roms/* $out/
                '';
              };

              fixAlePy =
                pkg:
                let
                  sdistPkg = pkg.override { sourcePreference = "sdist"; };
                in
                sdistPkg.overrideAttrs (old: {
                  nativeBuildInputs =
                    (old.nativeBuildInputs or [ ])
                    ++ [
                      pkgs.cmake
                      pkgs.pkg-config
                    ]
                    ++ (final.resolveBuildSystem {
                      "scikit-build-core" = [ ];
                      "nanobind" = [ ];
                    });
                  buildInputs = (old.buildInputs or [ ]) ++ [
                    pkgs.opencv
                    pkgs.SDL2
                    pkgs.sdl3
                    pkgs.zlib
                  ];
                  postInstall = ''
                    ${old.postInstall or ""}
                    romsDir="$out/${python.sitePackages}/ale_py/roms"
                    mkdir -p "$romsDir"
                    cp -r  ${ale-py-roms}/* "$romsDir/"
                  '';
                });
            in
            {
              numba = prev.numba.overrideAttrs (old: {
                buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.onetbb ];
              });
              torch =
                (hacks.nixpkgsPrebuilt {
                  from = pkgs.python314Packages.torchWithoutCuda;
                  prev = prev.torch;
                }).overrideAttrs
                  (old: {
                    passthru = (old.passthru or { }) // {
                      dependencies = lib.filterAttrs (
                        name: _: !(lib.hasPrefix "nvidia-" name || lib.hasPrefix "cuda-" name)
                      ) (old.passthru.dependencies or { });
                    };
                  });
              torchvision =
                (hacks.nixpkgsPrebuilt {
                  from = pkgs.python314Packages.torchvision;
                  prev = prev.torchvision;
                }).overrideAttrs
                  (old: {
                    passthru = (old.passthru or { }) // {
                      dependencies = lib.filterAttrs (
                        name: _: !(lib.hasPrefix "nvidia-" name || lib.hasPrefix "cuda-" name)
                      ) (old.passthru.dependencies or { });
                    };
                  });
            }
            // lib.optionalAttrs (prev ? "ale-py") { "ale-py" = fixAlePy prev."ale-py"; }
            // lib.optionalAttrs (prev ? "ale_py") { "ale_py" = fixAlePy prev."ale_py"; };
        in
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
          (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.wheel
              overlay
              customOverlay
            ]
          )
      );

    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          pythonSet = pythonSets.${system}.overrideScope editableOverlay;
          virtualenv = pythonSet.mkVirtualEnv "auga-dev-env" workspace.deps.all;

        in
        {
          default = pkgs.mkShell {
            packages = [
              virtualenv
              pkgs.SDL2
              pkgs.sdl3
              pkgs.uv
              pkgs.perf
              pkgs.lldb
              #              pkgs.nodejs_24
              pkgs.clinfo # GPU detection
              pkgs.opencl-headers
              pkgs.ocl-icd
              pkgs.intel-compute-runtime-legacy1
              (pkgs.writeShellScriptBin "jn" "exec jupyter notebook\"$@\"")
            ];
            env = {
              UV_NO_SYNC = "1";
              UV_PYTHON = pythonSet.python.interpreter;
              UV_PYTHON_DOWNLOADS = "never";
              PYSDL2_DLL_PATH = "${pkgs.SDL2}/lib";
              LD_LIBRARY_PATH = lib.makeLibraryPath [
                pkgs.SDL2
                pkgs.sdl3
              ];
            };
            shellHook = ''
              unset PYTHONPATH
              export REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
              if [ -f "$REPO_ROOT/token" ]; then
                export HF_TOKEN=$(cat "$REPO_ROOT/token")
              fi
              export PYTHONPATH="$REPO_ROOT/src"
            '';
          };
        }
      );

      packages = forAllSystems (system: {
        default = pythonSets.${system}.mkVirtualEnv "auga-env" workspace.deps.default;
      });
    };
}
