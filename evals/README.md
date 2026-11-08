# Evaluating the skills

Five fixed briefs in `briefs.json`, three against `japan` and two against
`abstract`, measure whether the plugin's skills make better moodboards. Each
brief runs twice in Claude Code, once with the `arcana` plugin installed
(skills and server) and once with the server alone.

## A run

1. Install the plugin from this repository:
   `/plugin marketplace add <path to this repo>`, then `/plugin install arcana@arcana`.
2. In a fresh session, for each brief, ask for the board with its id, for
   example: *Build a moodboard for "monsoon melancholy" from japan. Save it as
   eval-monsoon-skills and render it as HTML.*
3. Remove the plugin, keep the server (`claude mcp add arcana -- uv run
   --directory <repo> arcana-mcp`), and repeat with the
   `-plain` ids in fresh sessions.
4. Score the boards:

   ```
   python evals/score.py
   ```

5. Open each pair of HTML boards (`output/boards/eval-<id>-skills.html` and
   `-plain.html`) side by side without looking at which is which, and rank
   them. The ranking decides. The scores say whether each board meets the
   floor the moodboard skill sets: at least four covered angles, one hero, no
   near-duplicates, a palette, a reason for every section, derived pictures
   at most a third.

Angle coverage against each brief's listed angles is judged by the person
with the ranking, since a section's title rarely matches an angle word for
word.

## Before running

`japan` and `abstract` must point at their files. If `arcana dataset list`
says FILES MOVED, run `arcana-relocate` (or Relocate on the Datasets tab)
first, or the boards render without their pictures.
