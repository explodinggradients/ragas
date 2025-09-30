<!--
WHITELIST_AFTER_APPROVAL:
- Bash(git push:*)
- Bash(gh pr create:*)
-->

Please create a pull request with the changes on the current branch. $ARGUMENTS

**IMPORTANT: Use planning mode for this command.**

Follow these steps:

## Planning Phase (Research Only):
1. Check git status to ensure working tree is clean
2. Verify current branch is not main/master
3. Check if current branch tracks a remote branch and is up to date
4. Run `git log --oneline main..HEAD` to see commits that will be included in PR
5. Run `git diff main...HEAD` to understand the full scope of changes
6. Analyze all changes and commits to create a comprehensive PR summary
7. Draft PR title and description with:
   - Title: Clear, descriptive title based on the changes
   - Body: Include "## Summary" with bullet points, "## Test plan" checklist, and Claude signature
8. **Present the draft PR title and description using ExitPlanMode tool for user approval**

## Execution Phase (After User Approval):
9. Push branch to remote with upstream tracking if needed
10. Create PR using `gh pr create` with the approved title and description
11. Return the PR URL for easy access

Use the GitHub CLI (`gh`) for all GitHub-related operations. If $ARGUMENTS are provided, incorporate them as hints for the PR title or description.