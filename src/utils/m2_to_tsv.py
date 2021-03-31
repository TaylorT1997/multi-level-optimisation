import sys

def process(m2_file, errortypes):
    with open(m2_file, 'r') as fr:
        while True:
            sentence = None
            errors = None
            for line in fr:
                if line.startswith("S "):
                    sentence = line.strip().split()[1:]
                    errors = ["c" for i in range(len(sentence))]
                elif line.startswith("A "):
                    line = line[2:]
                    line_parts = line.split("|||")
                    index_from = int(line_parts[0].split()[0])
                    index_to = int(line_parts[0].split()[1])
                    if index_from == -1 and index_to == -1:
                        continue
                    if index_from == index_to:
                        index_to = index_to + 1
                    error_type = line_parts[1]
                    for i in range(index_from, min(index_to, len(sentence))):
                        if errortypes == True:
                            errors[i] = error_type
                        else:
                            errors[i] = "i"
                elif sentence == None and len(line.strip()) == 0:
                    continue
                elif sentence != None and len(line.strip()) == 0:
                    break
                else:
                    raise ValueError("Unknown format")
            if sentence != None:
                for i in range(len(sentence)):
                    print(sentence[i] + "\t" + errors[i])
                print("")
            else:
                break

if __name__ == "__main__":
    errortypes = True if sys.argv[1] == "true" else False
    inputfile = sys.argv[2]
    process(inputfile, errortypes)
