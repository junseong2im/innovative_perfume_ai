#!/usr/bin/env python3
"""
Release Notes Generator

Automatically generates release notes from git commits, issues, and pull requests.

Usage:
    python scripts/generate_release_notes.py --from v0.1.0 --to v0.2.0
    python scripts/generate_release_notes.py --auto  # Auto-detect last tag
    python scripts/generate_release_notes.py --from v0.1.0 --to HEAD --output RELEASE_NOTES.md
"""

import argparse
import subprocess
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys


# =============================================================================
# Git Operations
# =============================================================================

def run_git_command(command: List[str]) -> str:
    """Run git command and return output"""
    try:
        result = subprocess.run(
            ['git'] + command,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running git command: {e}", file=sys.stderr)
        return ""


def get_latest_tag() -> Optional[str]:
    """Get the latest git tag"""
    output = run_git_command(['describe', '--tags', '--abbrev=0'])
    return output if output else None


def get_commits_between(from_ref: str, to_ref: str) -> List[Dict[str, str]]:
    """Get commits between two references"""
    # Format: <hash>|<author>|<date>|<subject>|<body>
    format_str = '%H|%an|%ad|%s|%b'
    output = run_git_command([
        'log',
        f'{from_ref}..{to_ref}',
        f'--pretty=format:{format_str}',
        '--date=short'
    ])

    commits = []
    for line in output.split('\n\n'):
        if not line.strip():
            continue

        parts = line.split('|', 4)
        if len(parts) >= 4:
            commits.append({
                'hash': parts[0],
                'author': parts[1],
                'date': parts[2],
                'subject': parts[3],
                'body': parts[4] if len(parts) > 4 else ''
            })

    return commits


def get_current_branch() -> str:
    """Get current git branch"""
    return run_git_command(['rev-parse', '--abbrev-ref', 'HEAD'])


# =============================================================================
# Commit Classification
# =============================================================================

COMMIT_CATEGORIES = {
    'feat': 'Features',
    'fix': 'Bug Fixes',
    'perf': 'Performance Improvements',
    'docs': 'Documentation',
    'style': 'Code Style',
    'refactor': 'Code Refactoring',
    'test': 'Tests',
    'build': 'Build System',
    'ci': 'CI/CD',
    'chore': 'Chores',
    'security': 'Security',
    'infra': 'Infrastructure'
}


def parse_conventional_commit(subject: str) -> Tuple[str, str, Optional[str], bool]:
    """
    Parse conventional commit format: type(scope): description

    Returns: (type, description, scope, breaking)
    """
    # Pattern: type(scope)!: description or type!: description
    pattern = r'^(\w+)(\([^)]+\))?(!)?: (.+)$'
    match = re.match(pattern, subject)

    if match:
        commit_type = match.group(1).lower()
        scope = match.group(2).strip('()') if match.group(2) else None
        breaking = match.group(3) == '!'
        description = match.group(4)
        return commit_type, description, scope, breaking
    else:
        # Not conventional commit, categorize as 'other'
        return 'other', subject, None, False


def classify_commits(commits: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """Classify commits by category"""
    categorized = {category: [] for category in COMMIT_CATEGORIES.values()}
    categorized['Other'] = []
    breaking_changes = []

    for commit in commits:
        commit_type, description, scope, breaking = parse_conventional_commit(commit['subject'])

        # Check for breaking changes
        if breaking or 'BREAKING CHANGE' in commit['body']:
            breaking_changes.append({
                **commit,
                'description': description,
                'scope': scope
            })

        # Categorize commit
        category = COMMIT_CATEGORIES.get(commit_type, 'Other')
        categorized[category].append({
            **commit,
            'description': description,
            'scope': scope,
            'breaking': breaking
        })

    # Add breaking changes section if any
    if breaking_changes:
        categorized['Breaking Changes'] = breaking_changes

    # Remove empty categories
    return {k: v for k, v in categorized.items() if v}


# =============================================================================
# Release Notes Generation
# =============================================================================

def extract_issue_numbers(text: str) -> List[str]:
    """Extract issue/PR numbers from text"""
    # Pattern: #123, fixes #123, closes #123, etc.
    pattern = r'(?:fix(?:es)?|close(?:s)?|resolve(?:s)?|ref(?:s)?)?[\s#]*(\d+)'
    matches = re.findall(pattern, text.lower())
    return list(set(matches))


def format_commit_line(commit: Dict[str, str], show_author: bool = False) -> str:
    """Format a single commit line"""
    desc = commit['description']
    scope = f"**{commit['scope']}**: " if commit.get('scope') else ""
    author = f" (by @{commit['author']})" if show_author else ""
    hash_short = commit['hash'][:7]
    issues = extract_issue_numbers(commit['subject'] + ' ' + commit.get('body', ''))
    issue_links = ' '.join([f"[#{num}]" for num in issues])

    return f"- {scope}{desc} `{hash_short}`{author} {issue_links}"


def generate_release_notes(
    from_version: str,
    to_version: str,
    categorized_commits: Dict[str, List[Dict[str, str]]],
    output_file: Optional[Path] = None
) -> str:
    """Generate release notes markdown"""

    # Build release notes
    lines = [
        f"# Release Notes: {to_version}",
        f"",
        f"**Release Date**: {datetime.now().strftime('%Y-%m-%d')}",
        f"**Previous Version**: {from_version}",
        f"",
        "---",
        ""
    ]

    # Summary
    total_commits = sum(len(commits) for commits in categorized_commits.values())
    lines.append(f"## 📊 Summary")
    lines.append(f"")
    lines.append(f"This release includes **{total_commits} commits** across multiple categories.")
    lines.append(f"")

    # Breaking Changes (if any)
    if 'Breaking Changes' in categorized_commits:
        lines.append("## ⚠️ BREAKING CHANGES")
        lines.append("")
        lines.append("This release contains breaking changes. Please review carefully before upgrading.")
        lines.append("")
        for commit in categorized_commits['Breaking Changes']:
            lines.append(format_commit_line(commit, show_author=True))
        lines.append("")
        lines.append("---")
        lines.append("")

    # All changes by category
    lines.append("## 📦 Changes")
    lines.append("")

    for category, commits in categorized_commits.items():
        if category == 'Breaking Changes':
            continue  # Already handled above

        # Category emoji mapping
        emojis = {
            'Features': '✨',
            'Bug Fixes': '🐛',
            'Performance Improvements': '⚡',
            'Documentation': '📚',
            'Security': '🔒',
            'Infrastructure': '🔧',
            'Tests': '🧪'
        }
        emoji = emojis.get(category, '📝')

        lines.append(f"### {emoji} {category}")
        lines.append("")
        for commit in commits:
            lines.append(format_commit_line(commit))
        lines.append("")

    # Contributors
    lines.append("---")
    lines.append("")
    lines.append("## 👥 Contributors")
    lines.append("")
    authors = set(commit['author'] for commits in categorized_commits.values() for commit in commits)
    for author in sorted(authors):
        lines.append(f"- @{author}")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("## 🚀 Deployment")
    lines.append("")
    lines.append("See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for deployment instructions.")
    lines.append("")
    lines.append("## 🔄 Rollback")
    lines.append("")
    lines.append("If you need to rollback, see the rollback procedures in [DEPLOYMENT.md](docs/DEPLOYMENT.md).")
    lines.append("")
    lines.append(f"**Full Changelog**: {from_version}...{to_version}")

    content = '\n'.join(lines)

    # Write to file if specified
    if output_file:
        output_file.write_text(content, encoding='utf-8')
        print(f"✅ Release notes written to: {output_file}")

    return content


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate release notes from git commits',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--from',
        dest='from_ref',
        help='Starting git reference (tag, branch, commit)'
    )
    parser.add_argument(
        '--to',
        dest='to_ref',
        default='HEAD',
        help='Ending git reference (default: HEAD)'
    )
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Auto-detect from last tag to HEAD'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=Path,
        help='Output file path (default: print to stdout)'
    )
    parser.add_argument(
        '--show-authors',
        action='store_true',
        help='Show commit authors in changelog'
    )

    args = parser.parse_args()

    # Determine version range
    if args.auto:
        from_ref = get_latest_tag()
        if not from_ref:
            print("Error: No tags found. Cannot auto-detect version range.", file=sys.stderr)
            sys.exit(1)
        to_ref = 'HEAD'
        print(f"Auto-detected range: {from_ref}..{to_ref}")
    else:
        if not args.from_ref:
            print("Error: --from is required (or use --auto)", file=sys.stderr)
            sys.exit(1)
        from_ref = args.from_ref
        to_ref = args.to_ref

    # Get commits
    print(f"Fetching commits from {from_ref} to {to_ref}...")
    commits = get_commits_between(from_ref, to_ref)

    if not commits:
        print(f"No commits found between {from_ref} and {to_ref}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(commits)} commits")

    # Classify commits
    categorized = classify_commits(commits)

    # Generate release notes
    to_version = to_ref if to_ref != 'HEAD' else f"unreleased-{datetime.now().strftime('%Y%m%d')}"
    release_notes = generate_release_notes(
        from_version=from_ref,
        to_version=to_version,
        categorized_commits=categorized,
        output_file=args.output
    )

    # Print to stdout if no output file
    if not args.output:
        print("\n" + release_notes)


if __name__ == '__main__':
    main()
