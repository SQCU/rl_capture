#!/usr/bin/env python3
"""
OBS Studio Setup Script

Downloads and installs a pinned version of OBS Studio for reliable game capture.

Why OBS?
- Game Capture is BLESSED SOFTWARE - anti-cheat systems whitelist it
- No hooks/injections that trigger moderation APIs
- Window capture works by HWND/PID, not fragile title matching
- Virtual Camera output enables zero-copy frame sharing
- Mature, battle-tested capture pipeline

Pinned Version: 30.2.3 (stable, with Virtual Camera support)
"""

import os
import sys
import platform
import subprocess
import tempfile
import shutil
import hashlib
from pathlib import Path
from urllib.request import urlretrieve
from typing import Optional

# Pinned OBS version - change this to update
OBS_VERSION = "30.2.3"

# Download URLs for each platform (from official GitHub releases)
OBS_DOWNLOADS = {
    "Windows": {
        "url": f"https://github.com/obsproject/obs-studio/releases/download/{OBS_VERSION}/OBS-Studio-{OBS_VERSION}-Windows-Installer.exe",
        "sha256": None,  # Add hash after verification
        "installer": True,
    },
    "Windows-portable": {
        "url": f"https://github.com/obsproject/obs-studio/releases/download/{OBS_VERSION}/OBS-Studio-{OBS_VERSION}-Windows.zip",
        "sha256": None,
        "installer": False,
    },
    "Linux-ubuntu": {
        # Ubuntu/Debian uses the PPA
        "ppa": "ppa:obsproject/obs-studio",
        "package": "obs-studio",
    },
    "Linux-flatpak": {
        "flatpak_id": "com.obsproject.Studio",
    },
    "Darwin": {
        "url": f"https://github.com/obsproject/obs-studio/releases/download/{OBS_VERSION}/obs-studio-{OBS_VERSION}-macos-arm64.dmg",
        "sha256": None,
        "installer": True,
    },
    "Darwin-x86": {
        "url": f"https://github.com/obsproject/obs-studio/releases/download/{OBS_VERSION}/obs-studio-{OBS_VERSION}-macos-x86_64.dmg",
        "sha256": None,
        "installer": True,
    },
}

# Default install locations
INSTALL_PATHS = {
    "Windows": Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "obs-studio",
    "Windows-portable": Path("./obs-studio-portable"),
    "Linux": Path("/usr/bin/obs"),
    "Darwin": Path("/Applications/OBS.app"),
}


def get_platform_key() -> str:
    """Determine the platform key for downloads."""
    system = platform.system()
    if system == "Windows":
        return "Windows"
    elif system == "Linux":
        # Check if flatpak is available
        if shutil.which("flatpak"):
            return "Linux-flatpak"
        return "Linux-ubuntu"
    elif system == "Darwin":
        machine = platform.machine()
        if machine == "arm64":
            return "Darwin"
        return "Darwin-x86"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def check_obs_installed() -> Optional[Path]:
    """Check if OBS is already installed and return its path."""
    system = platform.system()

    if system == "Windows":
        # Check common install locations
        paths = [
            Path(os.environ.get("PROGRAMFILES", "")) / "obs-studio" / "bin" / "64bit" / "obs64.exe",
            Path(os.environ.get("PROGRAMFILES(X86)", "")) / "obs-studio" / "bin" / "64bit" / "obs64.exe",
            Path("./obs-studio-portable/bin/64bit/obs64.exe"),
        ]
        for p in paths:
            if p.exists():
                return p

        # Check PATH
        obs_path = shutil.which("obs64")
        if obs_path:
            return Path(obs_path)

    elif system == "Linux":
        # Check flatpak
        result = subprocess.run(
            ["flatpak", "list", "--app", "--columns=application"],
            capture_output=True, text=True
        )
        if "com.obsproject.Studio" in result.stdout:
            return Path("flatpak:com.obsproject.Studio")

        # Check system install
        obs_path = shutil.which("obs")
        if obs_path:
            return Path(obs_path)

    elif system == "Darwin":
        app_path = Path("/Applications/OBS.app")
        if app_path.exists():
            return app_path

    return None


def get_obs_version(obs_path: Path) -> Optional[str]:
    """Get the version of installed OBS."""
    try:
        if str(obs_path).startswith("flatpak:"):
            result = subprocess.run(
                ["flatpak", "info", "com.obsproject.Studio"],
                capture_output=True, text=True
            )
            for line in result.stdout.split("\n"):
                if "Version:" in line:
                    return line.split(":")[-1].strip()
        else:
            result = subprocess.run(
                [str(obs_path), "--version"],
                capture_output=True, text=True, timeout=5
            )
            # Parse version from output
            for line in result.stdout.split("\n"):
                if "OBS Studio" in line:
                    parts = line.split()
                    for part in parts:
                        if part[0].isdigit():
                            return part
    except Exception:
        pass
    return None


def download_file(url: str, dest: Path, expected_sha256: Optional[str] = None) -> bool:
    """Download a file with progress indicator and optional hash verification."""
    print(f"Downloading: {url}")
    print(f"Destination: {dest}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")

    try:
        urlretrieve(url, dest, reporthook=progress_hook)
        print()  # Newline after progress

        if expected_sha256:
            print("Verifying checksum...")
            sha256 = hashlib.sha256()
            with open(dest, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            if sha256.hexdigest() != expected_sha256:
                print(f"ERROR: Checksum mismatch!")
                print(f"  Expected: {expected_sha256}")
                print(f"  Got:      {sha256.hexdigest()}")
                return False
            print("Checksum OK")

        return True
    except Exception as e:
        print(f"ERROR: Download failed: {e}")
        return False


def install_obs_windows(portable: bool = False) -> bool:
    """Install OBS on Windows."""
    key = "Windows-portable" if portable else "Windows"
    config = OBS_DOWNLOADS[key]

    with tempfile.TemporaryDirectory() as tmpdir:
        if portable:
            # Download and extract portable version
            zip_path = Path(tmpdir) / "obs.zip"
            if not download_file(config["url"], zip_path, config.get("sha256")):
                return False

            print("Extracting portable OBS...")
            import zipfile
            dest = Path("./obs-studio-portable")
            dest.mkdir(exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(dest)
            print(f"Extracted to: {dest.absolute()}")

        else:
            # Download and run installer
            installer_path = Path(tmpdir) / "obs-installer.exe"
            if not download_file(config["url"], installer_path, config.get("sha256")):
                return False

            print("Running OBS installer...")
            print("NOTE: Please complete the installation wizard.")
            subprocess.run([str(installer_path)], check=True)

    return True


def install_obs_linux_flatpak() -> bool:
    """Install OBS on Linux via Flatpak."""
    print("Installing OBS via Flatpak...")
    try:
        subprocess.run(
            ["flatpak", "install", "-y", "flathub", "com.obsproject.Studio"],
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Flatpak install failed: {e}")
        return False


def install_obs_linux_apt() -> bool:
    """Install OBS on Ubuntu/Debian via apt."""
    print("Installing OBS via apt...")
    try:
        # Add PPA
        subprocess.run(
            ["sudo", "add-apt-repository", "-y", OBS_DOWNLOADS["Linux-ubuntu"]["ppa"]],
            check=True
        )
        # Update
        subprocess.run(["sudo", "apt", "update"], check=True)
        # Install
        subprocess.run(
            ["sudo", "apt", "install", "-y", OBS_DOWNLOADS["Linux-ubuntu"]["package"]],
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: apt install failed: {e}")
        return False


def install_obs_macos() -> bool:
    """Install OBS on macOS."""
    machine = platform.machine()
    key = "Darwin" if machine == "arm64" else "Darwin-x86"
    config = OBS_DOWNLOADS[key]

    with tempfile.TemporaryDirectory() as tmpdir:
        dmg_path = Path(tmpdir) / "obs.dmg"
        if not download_file(config["url"], dmg_path, config.get("sha256")):
            return False

        print("Mounting DMG...")
        mount_result = subprocess.run(
            ["hdiutil", "attach", str(dmg_path), "-mountpoint", "/Volumes/OBS"],
            capture_output=True, text=True
        )

        if mount_result.returncode != 0:
            print(f"ERROR: Failed to mount DMG: {mount_result.stderr}")
            return False

        try:
            print("Copying OBS.app to /Applications...")
            if Path("/Applications/OBS.app").exists():
                shutil.rmtree("/Applications/OBS.app")
            shutil.copytree("/Volumes/OBS/OBS.app", "/Applications/OBS.app")
        finally:
            subprocess.run(["hdiutil", "detach", "/Volumes/OBS"], check=False)

    return True


def setup_virtual_camera() -> bool:
    """Set up OBS Virtual Camera (required for frame capture)."""
    system = platform.system()

    if system == "Windows":
        print("\nVirtual Camera Setup (Windows):")
        print("  1. Open OBS Studio")
        print("  2. Go to Tools -> VirtualCam")
        print("  3. Click 'Install' if not already installed")
        print("  4. The virtual camera will be available as 'OBS Virtual Camera'")
        return True

    elif system == "Linux":
        print("\nVirtual Camera Setup (Linux):")
        print("  OBS uses v4l2loopback for virtual camera.")
        print("  Installing v4l2loopback-dkms...")
        try:
            subprocess.run(
                ["sudo", "apt", "install", "-y", "v4l2loopback-dkms"],
                check=True
            )
            # Load the module
            subprocess.run(
                ["sudo", "modprobe", "v4l2loopback", "video_nr=10",
                 "card_label='OBS Virtual Camera'", "exclusive_caps=1"],
                check=True
            )
            print("  Virtual camera available at /dev/video10")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: Failed to set up v4l2loopback: {e}")
            return False

    elif system == "Darwin":
        print("\nVirtual Camera Setup (macOS):")
        print("  Virtual Camera is built into OBS on macOS.")
        print("  Just click 'Start Virtual Camera' in OBS.")
        return True

    return False


def create_game_capture_scene(game_name: str, capture_method: str = "game") -> dict:
    """
    Create an OBS scene configuration for game capture.

    capture_method:
      - "game": Game Capture (Windows only, best for DirectX/OpenGL games)
      - "window": Window Capture (cross-platform, captures by HWND/PID)
      - "display": Display Capture (captures entire screen)

    Returns a scene configuration dict that can be imported into OBS.
    """
    scene_config = {
        "name": f"RL_Capture_{game_name.replace(' ', '_')}",
        "sources": []
    }

    if capture_method == "game" and platform.system() == "Windows":
        # Game Capture - uses HWND, not window title
        scene_config["sources"].append({
            "type": "game_capture",
            "name": f"{game_name}_GameCapture",
            "settings": {
                "capture_mode": "any_fullscreen",  # or "window" for specific window
                "capture_cursor": True,
                "anti_cheat_hook": True,  # Use anti-cheat compatible hook
                "allow_transparency": False,
                "priority": 1,  # Process priority matching
                "window": "",  # Will be set dynamically
            }
        })
    elif capture_method == "window":
        # Window Capture - works cross-platform
        scene_config["sources"].append({
            "type": "window_capture",
            "name": f"{game_name}_WindowCapture",
            "settings": {
                "cursor": True,
                "method": 0,  # 0 = Auto, 1 = BitBlt, 2 = WGC (Windows)
                "window": "",  # Format: "WindowTitle:WindowClass:Executable"
            }
        })
    else:
        # Display Capture - fallback
        scene_config["sources"].append({
            "type": "monitor_capture",
            "name": "DisplayCapture",
            "settings": {
                "cursor": True,
                "monitor": 0,
            }
        })

    return scene_config


def write_obs_scene_collection(scenes: list, output_path: Path):
    """Write an OBS scene collection JSON file."""
    import json

    collection = {
        "current_scene": scenes[0]["name"] if scenes else "Scene",
        "current_program_scene": scenes[0]["name"] if scenes else "Scene",
        "scene_order": [{"name": s["name"]} for s in scenes],
        "sources": [],
        "transitions": [],
    }

    # Flatten sources from all scenes
    for scene in scenes:
        for source in scene.get("sources", []):
            collection["sources"].append({
                "name": source["name"],
                "type": source["type"],
                "settings": source.get("settings", {}),
            })

    with open(output_path, "w") as f:
        json.dump(collection, f, indent=2)

    print(f"Scene collection written to: {output_path}")


def main():
    """Main setup routine."""
    import argparse

    parser = argparse.ArgumentParser(
        description="OBS Studio Setup for RL Capture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python obs_setup.py --check           # Check if OBS is installed
  python obs_setup.py --install         # Install OBS
  python obs_setup.py --portable        # Install portable version (Windows)
  python obs_setup.py --setup-vcam      # Set up Virtual Camera
  python obs_setup.py --scene "Sulfur"  # Generate scene config for game
        """
    )
    parser.add_argument("--check", action="store_true", help="Check OBS installation")
    parser.add_argument("--install", action="store_true", help="Install OBS")
    parser.add_argument("--portable", action="store_true", help="Use portable install (Windows)")
    parser.add_argument("--setup-vcam", action="store_true", help="Set up Virtual Camera")
    parser.add_argument("--scene", type=str, help="Generate scene config for game name")
    parser.add_argument("--capture-method", type=str, default="game",
                        choices=["game", "window", "display"],
                        help="Capture method for scene config")

    args = parser.parse_args()

    print("=" * 60)
    print(f"  OBS Studio Setup (Pinned Version: {OBS_VERSION})")
    print("=" * 60)
    print()

    # Check existing installation
    obs_path = check_obs_installed()
    if obs_path:
        version = get_obs_version(obs_path)
        print(f"OBS found: {obs_path}")
        print(f"Version: {version or 'unknown'}")

        if version and version != OBS_VERSION:
            print(f"WARNING: Installed version ({version}) differs from pinned ({OBS_VERSION})")
    else:
        print("OBS not found.")

    if args.check:
        sys.exit(0 if obs_path else 1)

    if args.install:
        if obs_path:
            print("\nOBS is already installed. Uninstall first to reinstall.")
            sys.exit(1)

        print(f"\nInstalling OBS {OBS_VERSION}...")
        platform_key = get_platform_key()

        success = False
        if platform_key.startswith("Windows"):
            success = install_obs_windows(portable=args.portable)
        elif platform_key == "Linux-flatpak":
            success = install_obs_linux_flatpak()
        elif platform_key == "Linux-ubuntu":
            success = install_obs_linux_apt()
        elif platform_key.startswith("Darwin"):
            success = install_obs_macos()

        if success:
            print("\nOBS installed successfully!")
        else:
            print("\nOBS installation failed.")
            sys.exit(1)

    if args.setup_vcam:
        setup_virtual_camera()

    if args.scene:
        scene = create_game_capture_scene(args.scene, args.capture_method)
        output_path = Path(f"./obs_scene_{args.scene.replace(' ', '_')}.json")
        write_obs_scene_collection([scene], output_path)

    print("\nSetup complete!")
    print("\nNext steps:")
    print("  1. Start OBS Studio")
    print("  2. Import the scene collection (if generated)")
    print("  3. Start Virtual Camera in OBS")
    print("  4. Run: python recording_stuff.py --obs")


if __name__ == "__main__":
    main()
