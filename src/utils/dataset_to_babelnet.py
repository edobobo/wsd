from nltk.corpus import wordnet as wn


def wn_sense_key_to_id(sense_key):
    synset = wn.lemma_from_key(sense_key).synset()
    return "wn:" + str(synset.offset()).zfill(8) + synset.pos()


_wn2bn = {}
_bn2wn = {}


with open("data/kb-mappings/bn2wn.txt") as f:
    for line in f:
        line = line.strip()
        parts = line.split("\t")
        _bn2wn[parts[0]] = parts[2]
        _wn2bn[parts[2]] = parts[0]


def wn_id2bn_id(wn_id):
    return _wn2bn[wn_id]


def bn_id2wn_id(bn_id):
    return _bn2wn[bn_id]


_to_bn_id_cache = {}


def to_bn_id(key):

    if key.startswith("bn:"):
        key_type = "bn_id"
        transform = lambda x: x
    elif key.startswith("wn:"):
        key_type = "wn_id"
        transform = lambda x: wn_id2bn_id(x)
    else:
        key_type = "sense_key"
        transform = lambda x: to_bn_id(wn_sense_key_to_id(x).replace("s", "a"))

    if key_type not in _to_bn_id_cache:
        _to_bn_id_cache[key_type] = {}

    if key not in _to_bn_id_cache[key_type]:
        _to_bn_id_cache[key_type][key] = transform(key)

    return _to_bn_id_cache[key_type][key]


def convert_to_bn(raganato_path: str, output_path: str) -> None:

    output_data_file_path = f"{output_path}.data.xml"
    output_key_file_path = f"{output_path}.gold.key.txt"

    print(output_data_file_path, output_key_file_path)

    output_key_file = open(output_key_file_path, "w")

    key_file_path = f"{raganato_path}.gold.key.txt"

    with open(key_file_path) as f:

        for kf_line in f:

            instance_id, *labels = kf_line.strip().split(" ")

            bn_synsets = []
            for label in labels:
                sense_synset = to_bn_id(label)
                if sense_synset not in bn_synsets:
                    bn_synsets.append(sense_synset)

            output_key_file.write(" ".join([instance_id] + bn_synsets) + "\n")

    output_key_file.close()

    data_file_path = f"{raganato_path}.data.xml"

    import os
    os.system(f"cp {data_file_path} {output_data_file_path}")


def main():
    import sys
    input_raganato_path = sys.argv[1]
    output_raganato_path = sys.argv[2]
    convert_to_bn(input_raganato_path, output_raganato_path)


if __name__ == '__main__':
    main()
