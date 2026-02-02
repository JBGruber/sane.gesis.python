"""Command-line interface for sane-gesis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sane_gesis.portable_repo import (
    build_portable_repo,
    export_install_script,
    plan_portable_repo,
)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="sane-gesis",
        description="Build portable Python package repositories for offline installation",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Plan command
    plan_parser = subparsers.add_parser(
        "plan",
        help="Scan a project to detect package dependencies",
    )
    plan_parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Directory to scan for Python files (default: current directory)",
    )
    plan_parser.add_argument(
        "-a", "--add",
        nargs="+",
        dest="add_pkgs",
        help="Additional packages to include",
    )
    plan_parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories",
    )
    plan_parser.add_argument(
        "-w", "--write-requirements",
        action="store_true",
        help="Write requirements.txt without prompting",
    )
    plan_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    # Build command
    build_parser = subparsers.add_parser(
        "build",
        help="Download packages and create a portable repository",
    )
    build_parser.add_argument(
        "-r", "--requirements",
        dest="requirements_file",
        help="Path to requirements.txt file",
    )
    build_parser.add_argument(
        "-p", "--packages",
        nargs="+",
        help="Package names to download",
    )
    build_parser.add_argument(
        "--platform",
        default="win_amd64",
        help="Target platform (default: win_amd64)",
    )
    build_parser.add_argument(
        "--python-version",
        default="3.11",
        help="Target Python version (default: 3.11)",
    )
    build_parser.add_argument(
        "-o", "--output",
        default="portable_repo.zip",
        help="Output zip file (default: portable_repo.zip)",
    )
    build_parser.add_argument(
        "--no-deps",
        action="store_true",
        help="Don't include dependencies",
    )
    build_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    # Export install script command
    export_parser = subparsers.add_parser(
        "export-script",
        help="Generate an offline installation script",
    )
    export_parser.add_argument(
        "-r", "--requirements",
        default="requirements.txt",
        dest="requirements_file",
        help="Requirements file path (default: requirements.txt)",
    )
    export_parser.add_argument(
        "-w", "--wheels-dir",
        default="wheels",
        help="Wheels directory name (default: wheels)",
    )
    export_parser.add_argument(
        "-o", "--output",
        default="install_offline.sh",
        help="Output script file (default: install_offline.sh)",
    )
    export_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    return parser


def cmd_plan(args: argparse.Namespace) -> int:
    """Execute the plan command."""
    pkgs = plan_portable_repo(
        path=args.path,
        add_pkgs=args.add_pkgs,
        recursive=not args.no_recursive,
        write_requirements=args.write_requirements if args.write_requirements else "ask",
        verbose=not args.quiet,
    )

    if not args.quiet:
        print("\nDetected packages:")
        for pkg in pkgs:
            print(f"  - {pkg}")

    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """Execute the build command."""
    # Get packages from requirements file or command line
    pkgs: list[str] = []

    if args.requirements_file:
        req_path = Path(args.requirements_file)
        if not req_path.exists():
            print(f"Error: Requirements file not found: {args.requirements_file}", file=sys.stderr)
            return 1
        pkgs = [
            line.strip()
            for line in req_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

    if args.packages:
        pkgs.extend(args.packages)

    if not pkgs:
        print("Error: No packages specified. Use -r or -p to specify packages.", file=sys.stderr)
        return 1

    build_portable_repo(
        pkgs=pkgs,
        platform=args.platform,
        python_version=args.python_version,
        out_file=args.output,
        include_deps=not args.no_deps,
        verbose=not args.quiet,
    )

    return 0


def cmd_export_script(args: argparse.Namespace) -> int:
    """Execute the export-script command."""
    export_install_script(
        requirements_file=args.requirements_file,
        wheels_dir=args.wheels_dir,
        script_file=args.output,
        verbose=not args.quiet,
    )
    return 0


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "plan":
        return cmd_plan(args)
    elif args.command == "build":
        return cmd_build(args)
    elif args.command == "export-script":
        return cmd_export_script(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
