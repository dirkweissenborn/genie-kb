"""
This script extracts facts in form of triples from UMLS RRF installation.
"""

import argparse
import os

parser = argparse.ArgumentParser(description='Extract triples from UMLS.')
parser.add_argument('--umls', help='directory containing UMLS RRF files.')
parser.add_argument('--vocabs', default='SNOMEDCT_US,MSH,CSP',
                    help='allowed comma separated source dictionaries (as used in MRCONSO), if empty use all')
parser.add_argument('--out', default='facts.tsv',
                    help='output file')


args = parser.parse_args()

vocabs = args.vocabs.split(",")

preferred = dict()

exclude_rels = ["mapped_to", "mapped_from",
                "has_mapping_qualifier", "mapping_qualifier_of",
                "moved_from", "moved_to",
                "has_associated_morphology", "associated_morphology_of",
                "classified_as",
                "possibly_equivalent_to"]

with open(args.out, 'w') as out:
    # Load allowed cuis and preferred labels
    with open(os.path.join(args.umls, "MRCONSO.RRF"), "r") as f:
        print("Processing MRCONSO...")
        for line in f:
            split = line.split("|")
            cui = split[0]
            lang = split[1]
            source = split[11]
            is_preferred = split[2] == "P"
            label = split[14]
            if lang == "ENG" and \
                    is_preferred and \
                    cui not in preferred and \
                    (not vocabs or source in vocabs):
                preferred[cui] = label
                out.write("%s\tpreferred_label\t%s\n" % (cui, label))

    with open(os.path.join(args.umls, "MRREL.RRF"), "r") as f:
        print("Processing MRREL...")
        for line in f:
            split = line.split("|")
            cui1 = split[0]
            cui2 = split[4]
            rel = split[7]
            if rel and rel not in exclude_rels and cui1 != cui2 and cui1 in preferred and cui2 in preferred:
                out.write("%s\t%s\t%s\n" % (cui2, rel, preferred[cui1]))

    with open(os.path.join(args.umls, "MRDEF.RRF"), "r") as f:
        print("Processing MRDEF...")
        for line in f:
            split = line.split("|")
            cui = split[0]
            definition = split[5]
            source = split[4]
            if cui in preferred and (not vocabs or source in vocabs):
                out.write("%s\tdefinition\t%s\n" % (cui, definition))



