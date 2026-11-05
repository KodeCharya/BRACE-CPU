import re
from collections import Counter

tex = open('paper/main.tex').read()
print("=== LaTeX syntax check ===")
print("\\begin{document}:", tex.count(r"\begin{document}"))
print("\\end{document}:", tex.count(r"\end{document}"))
print("\\maketitle:", tex.count(r"\maketitle"))

envs = re.findall(r"\\begin\{([^}]+)\}", tex)
end_envs = re.findall(r"\\end\{([^}]+)\}", tex)
env_count = Counter(envs)
end_count = Counter(end_envs)
all_envs = set(env_count) | set(end_count)
print("--- Environment balance ---")
all_balanced = True
for e in sorted(all_envs):
    b, e_ = env_count.get(e, 0), end_count.get(e, 0)
    status = "OK" if b == e_ else "MISMATCH"
    if b != e_:
        all_balanced = False
    print(f"  {e}: begin={b} end={e_} {status}")
print("Environments balanced:", all_balanced)

sections = re.findall(r"\\section\*?\{([^}]*)\}", tex)
print("--- Sections:", len(sections))
for s in sections:
    print("   ", s)

figs = re.findall(r"\\includegraphics\[([^\]]*)\]\{([^}]*)\}", tex)
print("--- Figures:", len(figs))
for o, p in figs:
    print("   ", p, "opts:", o)

tables = re.findall(r"\\begin\{tabular\}", tex)
print("--- tabular blocks:", len(tables))

cited = set()
for m in re.findall(r"\\citep\{([^}]*)\}", tex):
    for k in m.split(","):
        cited.add(k.strip())
print("--- Cited keys:", sorted(cited))

labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
refs = set(re.findall(r"\\ref\{([^}]*)\}", tex))
print("--- Labels:", sorted(labels))
print("--- Refs:", sorted(refs))
print("--- Undefined refs:", sorted(refs - labels))
print("--- Unused labels:", sorted(labels - refs))

bib_keys = set()
for line in open("paper/references.bib"):
    m = re.match(r"^@\w+\{([^,]+),", line)
    if m:
        bib_keys.add(m.group(1))
print("--- BibTeX keys:", sorted(bib_keys))
print("--- Cited but missing from bib:", sorted(cited - bib_keys))
print("=== Check complete ===")
