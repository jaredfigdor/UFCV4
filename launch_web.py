#!/usr/bin/env python3
"""
Launch UFC Predictions Web Interface
=====================================

Simple script to launch only the web interface without any
data scraping, dataset building, or model training.

Usage:
    python launch_web.py
    python launch_web.py --port 8080
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Launch the web interface only."""
    parser = argparse.ArgumentParser(description="Launch UFC Predictions Web Interface")
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for web interface (default: 5000)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open browser"
    )
    args = parser.parse_args()

    # Check if data folder exists
    data_folder = Path("ufc_data")
    if not data_folder.exists():
        logger.error("[ERROR] Data folder 'ufc_data' not found!")
        logger.error("Please run 'python app.py' first to generate predictions.")
        sys.exit(1)

    # Check if required files exist
    required_files = [
        "fight_predictions_summary.csv",
        "fighter_data.csv"
    ]

    missing_files = []
    for file in required_files:
        if not (data_folder / file).exists():
            missing_files.append(file)

    if missing_files:
        logger.warning(f"[WARNING] Missing files: {', '.join(missing_files)}")
        logger.warning("Some features may not work correctly.")
        logger.info("Run 'python app.py' to generate all data.")

    # Launch web interface
    logger.info("[WEB] Launching UFC Predictions Web Interface...")
    logger.info("=" * 50)

    try:
        from ufcscraper.web_app import launch_web_app

        launch_web_app(
            data_folder=data_folder,
            port=args.port,
            auto_open=not args.no_browser,
            debug=False
        )
    except KeyboardInterrupt:
        logger.info("\n[OK] Web interface closed")
    except Exception as e:
        logger.error(f"[ERROR] Error launching web interface: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
