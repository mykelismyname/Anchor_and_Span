import numpy as np
import pickle
import pandas as pd
import json
import argparse
import spacy
from scispacy.abbreviation import AbbreviationDetector
import scispacy
from scispacy.linking import EntityLinker

def load_model(model_path):
    ann_model = spacy.load(model_path)
    ann_model.add_pipe("abbreviation_detector")
    return ann_model

def annotate(proc_file, model):
    proc_file = pd.read_csv(proc_file, compression="gzip", low_memory=False)
    doc = model(proc_file["CLEANED_TEXT"][0])
    print(proc_file["CLEANED_TEXT"][0])
    for ent in doc.ents:
        print(ent)

def main(args):
    ann_model = load_model(args.model)
    ann_model.add_pipe("scispacy_linker", config={"linker_name": "umls"})
    annotate(proc_file=args.data, model=ann_model)

if __name__ == '__main__':
    par = argparse.ArgumentParser()
    par.add_argument('--data', default='../NOTESEVENTS_CLEANED.csv', help='location of the data')
    par.add_argument('--model', default='en_core_sci_sm', help='scispacy biomedical model')
    args = par.parse_args()
    main(args)
