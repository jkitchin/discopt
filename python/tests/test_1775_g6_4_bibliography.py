"""#1775 G6.4: the GDP literature of spec §11 is in ``docs/references.bib``.

The spec listed nine works "from memory, not verified; do not cite until
checked". Each was checked against its Crossref record before it was added; two
of the memory versions were wrong (LD-SDA is Ovalle et al. 2025, not Bernal et
al.; Pyomo.GDP is in Optimization and Engineering 23(1), 2022). This guards that
the keys exist, are unique, and carry the DOI they were verified by.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

BIB = Path(__file__).resolve().parents[2] / "docs" / "references.bib"

#: Key -> the DOI its Crossref record was checked under (#1775 G6.4).
VERIFIED = {
    "Lee2001": "10.1016/S0098-1354(01)00732-3",
    "Kronqvist2021": "10.1007/978-3-030-78230-6_19",
    "Kronqvist2025": "10.1007/s10107-025-02232-1",
    "Chen2022": "10.1007/s11081-021-09601-7",
    "Trespalacios2016": "10.1287/ijoc.2015.0669",
    "Ruiz2012": "10.1016/j.ejor.2011.10.002",
    "Ovalle2025": "10.1016/j.compchemeng.2024.108993",
    "Balas1985": "10.1137/0606047",
    "Balas1998": "10.1016/S0166-218X(98)00136-X",
}
#: The §11 keys that were already there.
ALREADY = [
    "Turkay1996",
    "Lee2000",
    "Trespalacios2015",
    "Furman2020",
    "Sawaya2005",
    "Grossmann2013",
    "Grossmann2002",
    "Balas1993",
    "RamanGrossmann1994",
]


def _entries() -> dict[str, str]:
    text = BIB.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for chunk in re.split(r"\n(?=@)", text):
        m = re.match(r"\s*@\w+\{([^,\s]+),", chunk)
        if m:
            assert m.group(1) not in out, f"duplicate key {m.group(1)}"
            out[m.group(1)] = chunk
    return out


def test_every_section_11_key_is_present():
    entries = _entries()
    missing = [k for k in [*VERIFIED, *ALREADY] if k not in entries]
    assert not missing, missing


@pytest.mark.parametrize("key, doi", sorted(VERIFIED.items()))
def test_each_added_key_carries_its_verified_doi(key, doi):
    entry = _entries()[key]
    assert entry.count("{") == entry.count("}"), key
    m = re.search(r"^\s*doi\s*=\s*\{([^}]*)\}", entry, re.M)
    assert m is not None and m.group(1).lower() == doi.lower(), (key, m and m.group(1))
    for field in ("author", "title", "year"):
        assert re.search(rf"^\s*{field}\s*=", entry, re.M), (key, field)
