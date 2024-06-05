lines2 = []
with open('../CLDM/data/chemformer_parsed2_shuffle.txt', 'r') as f:
    for _ in range(10000000):
        lines2.append(f.readline())

with open('./data/pubchem_10m.txt', 'w') as f:
    for l in lines2:
        f.write(l)
