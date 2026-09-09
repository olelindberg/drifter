# Writing style for `docs/`

These rules govern the prose in `docs/*.md`. They are descriptive of the existing documents —
match them rather than inventing a house style per file. They do not apply to code comments,
commit messages, or `README.md`.

## Two genres

**Theory documents** (`governing_equations.md`) derive physics from first principles and cite
no source files. Nothing in them refers to a class, a function, or a config key.

Write them as briefly as the mathematics allows. A section gives the setting, then the
equations, then the symbols, and stops.

- Assume nothing about what the equations are for. Do not say which terms a model would
  drop, which dominate, which are expensive, what a discretization will do with them, or how
  any of it will be solved. The document states the equations; it does not anticipate their
  use.
- Do not argue for the equations. No motivation for a step, no reason why a transformation
  is worth making, no physical interpretation of a term, no remark on what a result implies
  or why it matters. A derivation is a chain of statements, not a case for itself.
- Cut any sentence that could be deleted without losing a definition, an equation, or a
  condition under which one holds.
- This does not license stripping content. A convention, a definition, a named
  approximation, the frame a symbol belongs to, the range of a variable, and the conditions
  under which an equation holds are all part of the statement and stay. "$\theta$ is a
  colatitude, not a latitude" is a definition, not an explanation.

**Code-anchored derivations** (`cg_bezier_matrix_system.md`, `hermite_bathymetry_system.md`,
`coastline_adaptivity.md`) derive what the implementation actually assembles. In these, every
non-obvious claim carries a source link in the form
`[file.cpp:396-405](../src/bathymetry/file.cpp#L396-L405)`, and each opens with a
`**Scope:**` paragraph naming the classes covered. State plainly where documented behaviour
and assembled behaviour diverge — `cg_bezier_matrix_system.md` §10 is the model for this.

**Benchmark reports** (`hierarchical_ordering_benchmark.md`,
`uniform_vs_adaptive_convergence.md`) are short, table-led, and end in a conclusion that
names the decision the numbers support, including a negative one.

## Voice

- Write connected prose. Bullets are for enumerating symbols, options, or table-like facts —
  never for carrying an argument. A 700-line derivation should contain almost none.
- State results flatly, without hedging or self-congratulation: "No approximation has been
  made: these are the complete equations, and every term is retained."
- Introduce a sub-topic with a bold run-in lead-in — `**Metric.**`, `**Rotation.**`,
  `**Effective gravity.**` — in preference to a fourth heading level.
- Gloss every new symbol in prose immediately after the equation that introduces it: "Here
  $p$ is the pressure, $\rho$ the density and $g$ the magnitude of the effective gravity."
- Say when a convention is local to a section, and when it changes: "Within this section
  only, $x$, $y$, $z$ refer to the inertial frame."
- Address the reader through the material, not directly. No "we will now see", no "note that
  you should", no rhetorical questions.
- Do not close a section with a summary of what the section just said.

## Spelling and typography

- British spelling for `-our` and `-re`: *centre*, *behaviour*, *neighbour*.
- Oxford `-ize` for the verb suffix: *discretization*, *normalized*, *regularization*,
  *factorization*. (The documents currently hold roughly sixteen `-ise` stragglers against
  fifty `-ize`; write new text with `-ize` and leave existing text alone unless asked.)
- Real en dashes in compound names and ranges: Navier–Stokes, Bogner–Fox–Schmit,
  pressure–density coupling. Em dashes, unspaced, for parenthetical breaks.
- Wrap prose at roughly 95 columns.

## Mathematics

- Display equations in `$$ ... $$` on their own lines; inline maths in `$ ... $`.
- Punctuate display equations as part of the sentence that contains them — trailing `,` when
  the sentence continues, `.` when it ends.
- Number an equation with `\tag{n}` **only** when it is referred to by number elsewhere in
  the document. Most equations are not numbered. `governing_equations.md` is the exception:
  see below.
- **Equation numbering in `governing_equations.md`.** Every display equation in that document
  carries a `\tag{n}`, numbered consecutively from `(1)` in document order, with the tag on
  its own line between the equation's last line and the closing `$$`. The numbering is dense
  and total — no gaps, no unnumbered display equations, one number per `$$ ... $$` block even
  when the block co-displays several relations with `\qquad`. Adding, deleting or moving an
  equation therefore renumbers every equation after it: renumber the whole document in a
  single pass rather than patching locally, and re-check the prose cross-references, which
  cite equations in the bare form "the vertical momentum equation (34)".
- Use `\left( ... \right)` for delimiters around tall content, `\;` before units, and
  `\quad` / `\qquad` to separate co-displayed relations.
- Give a symbol table (`| Symbol | Meaning | Size | Built at |`) in code-anchored derivations
  where the count of matrices exceeds a handful.

## Structure

- One `#` title per document. In code-anchored derivations, title it
  `Subject — What Is Derived`, then open with a paragraph saying what the document derives
  and a `**Scope:**` paragraph.
- **Number the sections of every document**, theory documents included, so they can be
  cross-referenced as §7.1: `## 1. Overview` at the second level and `### 1.2 Subsection` at
  the third. `governing_equations.md` numbers both levels throughout; the code-anchored
  derivations number every `##` and those `###` that are cross-referenced. Adding, deleting
  or moving a section renumbers every section after it at that level — renumber in one pass
  and re-check the prose cross-references, which cite sections in the bare form "the rotated
  spherical frame of §1.4".
- `hermite_bathymetry_system.md` deliberately parallels the section numbering of
  `cg_bezier_matrix_system.md`. Preserve that correspondence when editing either.
