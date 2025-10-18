#!/usr/bin/env python3
"""
Verification script for UFC Web Interface installation.

Run this to check that all files are in place and dependencies are installed.
"""

import sys
from pathlib import Path


def check_files():
    """Check that all required files exist."""
    print("[*] Checking file structure...")

    required_files = [
        "ufcscraper/web_app.py",
        "ufcscraper/templates/base.html",
        "ufcscraper/templates/dashboard.html",
        "ufcscraper/templates/model_performance.html",
        "ufcscraper/templates/fight_detail.html",
        "ufcscraper/static/css/style.css",
        "ufcscraper/static/js/charts.js",
    ]

    all_present = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  [OK] {file_path}")
        else:
            print(f"  [MISSING] {file_path}")
            all_present = False

    return all_present


def check_dependencies():
    """Check that required Python packages are installed."""
    print("\n[*] Checking dependencies...")

    dependencies = {
        "flask": "Flask",
        "plotly": "Plotly",
        "pandas": "Pandas",
        "numpy": "NumPy",
        "sklearn": "scikit-learn",
    }

    all_installed = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [NOT INSTALLED] {name}")
            all_installed = False

    return all_installed


def check_app_modification():
    """Check that app.py has been modified with Step 7."""
    print("\n[*] Checking app.py modifications...")

    app_file = Path("app.py")
    if not app_file.exists():
        print("  [NOT FOUND] app.py")
        return False

    content = app_file.read_text(encoding='utf-8')

    checks = [
        ("--no-web flag", "--no-web"),
        ("--port flag", "--port"),
        ("Step 7 comment", "Step 7"),
        ("web_app import", "from ufcscraper.web_app import launch_web_app"),
        ("launch_web_app call", "launch_web_app("),
    ]

    all_present = True
    for check_name, check_str in checks:
        if check_str in content:
            print(f"  [OK] {check_name}")
        else:
            print(f"  [MISSING] {check_name}")
            all_present = False

    return all_present


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("UFC Web Interface - Installation Verification")
    print("=" * 60)
    print()

    files_ok = check_files()
    deps_ok = check_dependencies()
    app_ok = check_app_modification()

    print()
    print("=" * 60)

    if files_ok and deps_ok and app_ok:
        print("[SUCCESS] ALL CHECKS PASSED!")
        print()
        print("You're ready to go! Run:")
        print("  python app.py")
        print()
        print("Then visit http://localhost:5000 to see your predictions!")
        print("=" * 60)
        return 0
    else:
        print("[FAILED] SOME CHECKS FAILED")
        print()
        if not files_ok:
            print("[!] Some files are missing. Re-run the setup.")
        if not deps_ok:
            print("[!] Install missing dependencies:")
            print("   poetry install")
            print("   OR")
            print("   pip install flask plotly")
        if not app_ok:
            print("[!] app.py modifications incomplete. Check the file.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
