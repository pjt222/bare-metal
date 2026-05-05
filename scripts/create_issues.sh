#!/usr/bin/env bash
# scripts/create_issues.sh — Bulk-create GitHub issues from .github/issues/*.md
#
# Usage:
#   bash scripts/create_issues.sh         # create all issues
#   bash scripts/create_issues.sh 01 07   # create only #01 and #07
#
# Each .md file must have YAML frontmatter with:
#   title: "Issue title"
#   labels: ["label1", "label2"]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ISSUES_DIR="$REPO_ROOT/.github/issues"

if ! command -v gh &>/dev/null; then
    echo "ERROR: gh (GitHub CLI) not found. Install: https://cli.github.com/"
    exit 1
fi

if ! gh auth status &>/dev/null; then
    echo "ERROR: Not authenticated with gh. Run: gh auth login"
    exit 1
fi

create_issue() {
    local file="$1"
    local basename
    basename=$(basename "$file" .md)

    # Parse title from YAML frontmatter
    local title
    title=$(grep -m1 '^title:' "$file" | sed 's/^title: *"//' | sed 's/" *$//')
    if [[ -z "$title" ]]; then
        echo "WARN: skipping $basename — no title found"
        return
    fi

    # Parse labels from YAML frontmatter
    local labels=""
    local raw_labels
    raw_labels=$(grep -m1 '^labels:' "$file" | sed 's/^labels: *//' || true)
    if [[ -n "$raw_labels" ]]; then
        # Convert ["a", "b"] → a,b
        labels=$(echo "$raw_labels" | tr -d '[] ' | tr -d '"')
    fi

    # Body is everything after the frontmatter (after second ---)
    local body_file
    body_file="/tmp/gh_issue_body_${basename}.md"
    awk 'BEGIN{sep=0} /^---$/{sep++; next} sep>=2{print}' "$file" > "$body_file"

    echo "Creating issue $basename: $title"
    local created=false
    if [[ -n "$labels" ]]; then
        if gh issue create --title "$title" --body-file "$body_file" --label "$labels" 2>/dev/null; then
            created=true
        else
            echo "  ⚠ Label(s) [$labels] not found on repo, creating without labels"
        fi
    fi
    if [[ "$created" == false ]]; then
        gh issue create --title "$title" --body-file "$body_file"
    fi

    rm -f "$body_file"
    echo "  ✓ Created"
    sleep 1  # rate limit safety
}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if [[ $# -eq 0 ]]; then
    # All issues
    for f in "$ISSUES_DIR"/*.md; do
        [[ -f "$f" ]] || continue
        create_issue "$f"
    done
else
    # Specific issues by number prefix
    for num in "$@"; do
        # Pad to match filename pattern (01, 02, etc.)
        # Accept both "1" and "01"
        padded=$(printf "%02d" "${num#0}")
        file=$(find "$ISSUES_DIR" -maxdepth 1 -name "${padded}-*.md" | head -n1)
        if [[ -n "$file" && -f "$file" ]]; then
            create_issue "$file"
        else
            echo "WARN: issue file for #$num not found"
        fi
    done
fi

echo "Done."
