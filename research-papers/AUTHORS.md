# Authors — canonical block for all papers in `research-papers/`

Use this file as the single source of truth for the byline, affiliations and ORCIDs.
Do **not** invent ORCIDs, e-mail addresses or phone numbers; if a journal asks for a
field that is not recorded here, ask the authors rather than filling a placeholder.
(The TJEECS draft once shipped with ORCID's own documentation example ID — that is the
failure mode this file exists to prevent.)

## Byline order

1. **Anandkumar Pardeshi** — corresponding author (`*`)
2. **Sujata Deshmukh**

## Author 1 — Anandkumar Pardeshi

| Field | Value |
|---|---|
| Name (journal caps style) | Anandkumar PARDESHI |
| Name (normal style) | Anandkumar Pardeshi |
| Role | Corresponding author |
| Affiliation | Department of Computer Science and Engineering, Fr. C. Rodrigues Institute of Technology, University of Mumbai |
| Campus / city | Vashi, Navi Mumbai, Maharashtra, India |
| ORCID | 0000-0003-2806-3305 — <https://orcid.org/0000-0003-2806-3305> |
| E-mail | `anand.pardeshi@fcrit.ac.in` (author-confirmed 2026-08-17) |
| Phone | *not supplied by the author; do not invent one* |

## Author 2 — Sujata Deshmukh

| Field | Value |
|---|---|
| Name (journal caps style) | Sujata DESHMUKH |
| Name (normal style) | Sujata Deshmukh |
| Affiliation | Department of Computer Engineering, Fr. C. Rodrigues College of Engineering, University of Mumbai |
| Campus / city | Bandra, Mumbai, Maharashtra, India |
| ORCID | 0000-0001-9893-6947 — <https://orcid.org/0000-0001-9893-6947> |
| E-mail | `sujata.deshmukh@fragnel.edu.in` (author-confirmed 2026-08-17) |

## Provenance

The affiliations and ORCIDs above were supplied directly by the author on
2026-08-17 and confirmed as correct. Treat them as authoritative and do not
"improve" them against earlier drafts.

**The two authors are on different e-mail domains**, because they are at different
institutes: author 1 at `fcrit.ac.in` (Fr. C. Rodrigues Institute of Technology),
author 2 at `fragnel.edu.in` (Fr. C. Rodrigues College of Engineering, which
brands itself Fragnel). Never reconstruct one address from the other's pattern —
guessing `sujata.deshmukh@fcrit.ac.in` from author 1's domain gives the wrong
address.

## Notes and corrections to earlier drafts

- The two authors are at **different institutes**: author 1 at Fr. C. Rodrigues
  *Institute of Technology* (Vashi, Navi Mumbai), author 2 at Fr. C. Rodrigues
  *College of Engineering* (Bandra, Mumbai). Several earlier drafts
  (`research-paper-16-ijase-not-sent/ijase_mig.tex`, the TJEECS version) put both
  authors at the Vashi institute — that is wrong; use the table above.
- Both authors are affiliated to the **University of Mumbai**; include it when the
  journal's affiliation style allows a parent-university line.
- The e-mail `anand.pardeshi@fcrit.ac.in` and the phone `+91-22-2768-0000` appear in
  the IJASE and paper-5 drafts. Neither was supplied by the author and neither is
  confirmed; the phone in particular looks like a switchboard placeholder. **Do not
  carry either forward into a new paper without asking.**

## Ready-to-paste blocks

### LaTeX — Springer Nature (`sn-jnl.cls`, used by Financial Innovation)

```latex
\author*[1]{\fnm{Anandkumar} \sur{Pardeshi}}\email{...}
\author[2]{\fnm{Sujata} \sur{Deshmukh}}\email{...}

\affil*[1]{\orgdiv{Department of Computer Science and Engineering},
  \orgname{Fr. C. Rodrigues Institute of Technology, University of Mumbai},
  \orgaddress{\city{Vashi, Navi Mumbai}, \state{Maharashtra}, \country{India}}}

\affil[2]{\orgdiv{Department of Computer Engineering},
  \orgname{Fr. C. Rodrigues College of Engineering, University of Mumbai},
  \orgaddress{\city{Bandra, Mumbai}, \state{Maharashtra}, \country{India}}}
```

### LaTeX — plain `article` byline

```latex
\author{
Anandkumar Pardeshi\,$^{1,*}$ \quad and \quad Sujata Deshmukh\,$^{2}$\\[0.5em]
\small $^{1}$Department of Computer Science and Engineering,
Fr.\ C.\ Rodrigues Institute of Technology, University of Mumbai,\\
\small Vashi, Navi Mumbai, Maharashtra, India\\[0.2em]
\small $^{2}$Department of Computer Engineering,
Fr.\ C.\ Rodrigues College of Engineering, University of Mumbai,\\
\small Bandra, Mumbai, Maharashtra, India
}
```

### Plain text (submission forms)

```
Anandkumar PARDESHI (corresponding author)
Department of Computer Science and Engineering
Fr. C. Rodrigues Institute of Technology, University of Mumbai
Vashi, Navi Mumbai, India
ORCID: 0000-0003-2806-3305

Sujata DESHMUKH
Department of Computer Engineering
Fr. C. Rodrigues College of Engineering, University of Mumbai
Bandra, Mumbai, India
ORCID: 0000-0001-9893-6947
```
