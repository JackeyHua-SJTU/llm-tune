with open("data/dataset.txt", "r", encoding="utf-8") as fp:
    data = fp.read().strip().split("\n")
sentences = []

for d in data:
    d = d.strip()
    if "===" in d or len(d) == 0 or d == "aaa":
        continue
    sentences.append(d)

with open("data/processed_version.txt", "w", encoding="utf-8") as fp:
    fp.write("\n".join(sentences))