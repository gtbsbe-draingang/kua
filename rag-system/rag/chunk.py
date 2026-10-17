import re

# A numbered clause ("12. ..." / "12.3. ...") at line start always opens a new chunk,
# so each chunk carries the clause it came from and citations stay exact.
CLAUSE = re.compile(r"^(\d+(?:\.\d+)*)\.\s")
SPLIT = re.compile(r"\n\s*\n|\n(?=\s*\d+(?:\.\d+)*\.\s)")


def chunk(text: str, max_chars: int = 1500) -> list[dict]:
    out: list[dict] = []
    for para in (p.strip() for p in SPLIT.split(text)):
        if not para:
            continue
        m = CLAUSE.match(para)
        if m or not out or len(out[-1]["text"]) + len(para) > max_chars:
            clause = f"п. {m[1]}" if m else (out[-1]["clause"] if out else "")
            out.append({"clause": clause, "text": para})
        else:
            out[-1]["text"] += "\n\n" + para
    # hard-split anything still too long (one huge paragraph)
    return [{"clause": c["clause"], "text": c["text"][i:i + max_chars]}
            for c in out for i in range(0, len(c["text"]), max_chars)]


if __name__ == "__main__":
    doc = "Глава 1. Общие положения\n\n1. Первый пункт.\nпродолжение\n2.1. Подпункт.\n\nабзац к 2.1\n\n" + "x" * 3100
    cs = chunk(doc)
    assert cs[0] == {"clause": "", "text": "Глава 1. Общие положения"}, cs[0]
    assert cs[1] == {"clause": "п. 1", "text": "1. Первый пункт.\nпродолжение"}, cs[1]
    assert cs[2] == {"clause": "п. 2.1", "text": "2.1. Подпункт.\n\nабзац к 2.1"}, cs[2]
    assert [len(c["text"]) for c in cs[3:]] == [1500, 1500, 100] and cs[3]["clause"] == "п. 2.1"
    print("ok")
