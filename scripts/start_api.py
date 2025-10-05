#!/usr/bin/env python3
"""
Startup script for AgentZero API server.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # ← ADD THIS
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))



import uvicorn
import argparse
import logging

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Start AgentZero API server")
    
    parser.add_argument(
        "--host",
        default=os.getenv("API_HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("API_PORT", "8000")),
        help="Port to bind to (default: 8000)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("API_WORKERS", "1")),
        help="Number of worker processes (default: 1)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "info"),
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)"
    )
    
    return parser.parse_args()


def main():
    """Main startup function."""
    args = parse_args()
    
    print("=" * 60)
    print("AgentZero Prompt Injection Detector API")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Workers: {args.workers}")
    print(f"Reload: {args.reload}")
    print(f"Log Level: {args.log_level}")
    print("=" * 60)
    
    # Start server
    uvicorn.run(
        "src.api.server:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_level=args.log_level,
        access_log=True
    )


if __name__ == "__main__":
    main()