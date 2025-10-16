#!/usr/bin/env python3
"""
CLI module entry point for MATLAB2C++ Agentic Service.
This allows running the CLI as: python -m matlab2cpp_agentic_service.cli
"""

from .commands import cli

if __name__ == "__main__":
    cli()




