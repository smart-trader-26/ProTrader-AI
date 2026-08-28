"""Render refs.bib in IJCISIM's numbered house style.

IJCISIM numbers references in order of first appearance and prints them as

    12. Surname, Given, and Surname, Given, "Title of the work", Journal Name,
        vol. 24, no. 5, pp. 1481-1512, 2011.

which is what the published articles carry.  Building the list from the same
refs.bib the LaTeX sibling of this paper uses keeps one bibliography for both
manuscripts: an entry cannot be real in one and invented in the other, and
``verify_refs.py`` checks the single file by DOI.
"""
from __future__ import annotations

import re
from pathlib import Path

BIB = Path(__file__).resolve().parent.parent / "refs.bib"


def parse_bib(path: Path = BIB) -> dict:
    text = path.read_text(encoding="utf-8")
    out = {}
    for block in re.finditer(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", text, re.S):
        kind, key, bodytext = block.group(1), block.group(2).strip(), block.group(3)
        fields = {}
        for m in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*(?=\n\s*\w+\s*=|\Z)",
                             bodytext, re.S):
            fields[m.group(1).lower()] = " ".join(m.group(2).split())
        fields["_kind"] = kind.lower()
        out[key] = fields
    return out


def _authors(raw: str) -> str:
    """`Tetlock, Paul C. and Saar-Tsechansky, Maytal` -> IJCISIM name run."""
    raw = _strip_braces(raw)
    people = [a.strip() for a in re.split(r"\s+and\s+", raw) if a.strip()]
    names = []
    for p in people:
        if "," in p:
            last, first = [x.strip() for x in p.split(",", 1)]
            names.append(f"{last}, {first}")
        else:
            bits = p.split()
            names.append(f"{bits[-1]}, {' '.join(bits[:-1])}" if len(bits) > 1 else p)
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]}, and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


# refs.bib is shared with the LaTeX sibling of this paper, so it carries TeX
# escapes.  Word wants the characters themselves.
ACCENTS = {
    '"': {"a": "ä", "e": "ë", "i": "ï", "o": "ö", "u": "ü",
          "A": "Ä", "O": "Ö", "U": "Ü"},
    "'": {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú",
          "A": "Á", "E": "É", "I": "Í", "O": "Ó", "U": "Ú", "c": "ć"},
    "`": {"a": "à", "e": "è", "i": "ì", "o": "ò", "u": "ù"},
    "^": {"a": "â", "e": "ê", "i": "î", "o": "ô", "u": "û"},
    "~": {"n": "ñ", "a": "ã", "o": "õ"},
    "c": {"c": "ç", "C": "Ç", "s": "ş"},
    "v": {"c": "č", "s": "š", "z": "ž"},
}
TEX = re.compile(r"\{?\\([\"'`^~cv])\s*\{?([A-Za-z])\}?\}?")


def detex(s: str) -> str:
    s = TEX.sub(lambda m: ACCENTS.get(m.group(1), {}).get(m.group(2), m.group(2)), s)
    for tex, plain in (("\\&", "&"), ("\\%", "%"), ("\\$", "$"),
                       ("\\_", "_"), ("\\#", "#"), ("--", "-"), ("~", " ")):
        s = s.replace(tex, plain)
    return s


def _strip_braces(s: str) -> str:
    return detex(s).replace("{", "").replace("}", "")


def format_entry(e: dict) -> str:
    """One reference, in the journal's numbered style."""
    parts = [_authors(e.get("author", e.get("editor", "Anon.")))]
    title = _strip_braces(e.get("title", ""))
    parts.append(f'"{title}"')

    kind = e["_kind"]
    if kind == "article":
        venue = _strip_braces(e.get("journal", ""))
    elif kind in ("inproceedings", "incollection"):
        venue = _strip_braces(e.get("booktitle", ""))
    else:
        venue = _strip_braces(e.get("publisher", ""))
    if venue:
        parts.append(venue)

    if e.get("volume"):
        parts.append(f"vol. {e['volume']}")
    if e.get("number"):
        parts.append(f"no. {e['number']}")
    if e.get("pages"):
        parts.append("pp. " + e["pages"].replace("--", "-"))
    if kind == "book" and e.get("address"):
        parts.append(e["address"])
    parts.append(e.get("year", "n.d."))

    line = ", ".join(p for p in parts if p) + "."
    if e.get("doi"):
        line += f" doi:{e['doi']}."
    return line


class Bibliography:
    """Assigns numbers in order of first citation and renders the final list."""

    def __init__(self, path: Path = BIB):
        self.entries = parse_bib(path)
        self.order: list[str] = []

    def cite(self, *keys: str) -> str:
        """Return the bracketed marker for one or more keys, e.g. `[3, 7]`."""
        nums = []
        for k in keys:
            if k not in self.entries:
                raise KeyError(f"{k} is not in refs.bib")
            if k not in self.order:
                self.order.append(k)
            nums.append(self.order.index(k) + 1)
        return "[" + ", ".join(str(n) for n in nums) + "]"

    def rendered(self) -> list[str]:
        return [format_entry(self.entries[k]) for k in self.order]

    def unused(self) -> list[str]:
        return [k for k in self.entries if k not in self.order]


if __name__ == "__main__":
    bib = parse_bib()
    print(f"{len(bib)} entries in {BIB}")
    for key in list(bib)[:5]:
        print(" ", format_entry(bib[key]))
