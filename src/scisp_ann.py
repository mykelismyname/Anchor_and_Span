import numpy as np
import pickle
import pandas as pd
import json
import logging
import argparse
import spacy
import sys
import os
from scispacy.abbreviation import AbbreviationDetector
import scispacy
from scispacy.linking import EntityLinker
from glob import glob
from copy import deepcopy

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

#load spacy model
def load_model(model_path):
    ann_model = spacy.load(model_path)
    ann_model.add_pipe("abbreviation_detector")
    return ann_model

def annotate_entities(proc_file, dest, model):
    '''
    :param proc_file: file to annotate
    :param model: spacy model to perform entity linking
    :return: annotated file with token level annotations of entities and sentence level annotations of evidence
    '''
    proc_file = pd.read_csv(proc_file, low_memory=False)
    linker = model.get_pipe("scispacy_linker")
    ann_window = args.annotation_size.split()
    start, end = int(ann_window[0]), int(ann_window[1])
    data = proc_file["CLEANED_TEXT"].tolist()
    if end > -1:
        data = data[start:end]

    logging.info("The dataset is of size {}".format(len(data)))
    try:
        # docs = list(model.pipe(data))
        dataset_ann = []
        with open(dest+'/umls_anns_{}_{}.pkl'.format(start, end), 'wb') as umls_writer:
            for d in data:
                doc = model(d)
                doc_ann = {}
                # doc_sents = []
                sentences = []
                sent_entities = []
                for sent_id,sent in enumerate(doc.sents):
                    for entity in sent.ents:
                        entity_annotation = {"name": "{}".format(entity),
                                             "pos": [entity.start, entity.end],
                                             "sent_id": sent_id,
                                             "linked_umls_entities": []}
                        for umls_ent in entity._.kb_ents:
                            linked_entity = {}
                            umls_ent_detail = linker.kb.cui_to_entity[umls_ent[0]]
                            linked_entity["type_id"] = umls_ent_detail.types[0]
                            linked_entity["score"] = np.round(umls_ent[1],4)
                            linked_entity["type"] = linker.kb.semantic_type_tree.get_canonical_name(umls_ent_detail.types[0])
                            # linked_entity["aliases"] = umls_ent_detail.aliases
                            entity_annotation["linked_umls_entities"].append(linked_entity)
                        sent_entities.append([entity_annotation])
                    # doc_sents.append({"sent_pos": [sent.start, sent.end]})
                    sentences.append([m.text for m in sent])
                # doc_sents.append(sent_entities)
                # dataset_ann.append(doc_sents)
                doc_ann["Entities"] = sent_entities
                doc_ann["Sents"] = sentences
                dataset_ann.append(doc_ann)
            pickle.dump(dataset_ann, umls_writer, protocol=pickle.HIGHEST_PROTOCOL)

        # print(dataset_ann)
        logging.info("Number of documents annotated {}".format(len(dataset_ann)))
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        raise NotImplementedError("Entity linking with the UMLS KB wasn't successfully implemented")

def annotate_relations(data):
    files = glob(data+'/*.pkl')
    for file in files:
        with open(file, 'rb') as rf, open("test.json", 'wb') :
            entity_annotations = pickle.load(rf)
            test = []

            for ann in entity_annotations:
                #sort entities by sentence
                labels = []
                sentences = list(sorted(ann['Entities'], key=lambda x:x[0]['sent_id']))
                sentences_ = deepcopy(sentences)
                for e in range(len(sentences)):
                    entity_type = sentences[e][0]['linked_umls_entities'][0]
                    l,r = e,e
                    if len(entity_type) > 0:
                        if entity_type[0]['type'].lower() == 'clinical drug':
                            #search to the left and to the right
                            while l >= 0 or r < len(sentences):
                                if sentences[l][0]['linked_umls_entities'][0]['type'].lower() == 'Sign or Symptom':
                                    labels.append({'r':'is_treated_by', 'h':l, 't':e})
                                elif sentences[r][0]['linked_umls_entities'][0]['type'].lower() == 'Sign or Symptom':
                                    labels.append({'r': 'is_treated_by', 'h': l, 't': e})
                                l -= 1
                                r += 1
                sentences_['label'] = labels

        break

def main(args):
    annotate_type = args.annotate_type.split()
    if 'entities' in annotate_type:
        ann_model = load_model(args.model)
        ann_model.add_pipe("scispacy_linker", config={"linker_name": "umls"})
        #python scisp_ann.py --data ../NOTESEVENTS_CLEANED.csv --annotation_size "20000 22000"
        annotate_entities(proc_file=args.data, model=ann_model, dest=args.dest)
    if 'relations' in annotate_type:
        annotate_relations(args.data)

if __name__ == '__main__':
    par = argparse.ArgumentParser()
    par.add_argument('--data', default='../NOTESEVENTS_CLEANED.csv', help='location of the data')
    par.add_argument('--dest', default='../anns/spacy', help='location of the data')
    par.add_argument('--annotation_size', type=str, help='number of documents to annotate')
    par.add_argument('--model', default='en_core_sci_sm', help='scispacy biomedical model')
    par.add_argument('--annotate_type', default='entities', help="what to annotate 'entities', or 'relations' or 'entities relations'")
    par.add_argument('--kg', type=str, default='umls', help='which knoiwledge graph is linked to for annotation')
    args = par.parse_args()
    main(args)