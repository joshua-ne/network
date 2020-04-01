from find_all_path import *


G, hosts, fogs = build_graph("p2p-Gnutella04.txt")

for maxLen in range(1, 15):
    count = 2
    total = count;
    # print()
    # print("Trying to build all path with maxLength of " + str(maxLen) + "...", flush = True)
    start = time.time()
    all_path = {}
    for h in hosts:
        for f in fogs:
            all_path[(h,f)] = generate_path_set(h, f, G, maxLength = maxLen, silent = False)
            count = count - 1
            if count <= 0:
                break

        if count <= 0:
            break
    print(maxLen, (time.time() - start) / total, flush = True)