import sys

default_label = sys.argv[1]

with open(sys.argv[2], 'r') as f:
    words = []
    for line in f:
        if len(line.strip()) == 0:
            assert(len(words) > 0)
            sentence_label = default_label
            for w in words:
                if w[-1] != default_label:
                    sentence_label = w[-1]
            for w in words:
                print("\t".join(w[:-1] + [sentence_label]))
            print("")
            words = []
        else:
            words.append(line.strip().split())

    assert(len(words) == 0)
