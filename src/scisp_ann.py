import numpy as np
import pickle
import pandas as pd
import json
import argparse
import spacy
from scispacy.abbreviation import AbbreviationDetector
import scispacy
from scispacy.linking import EntityLinker

#load spacy model
def load_model(model_path):
    ann_model = spacy.load(model_path)
    ann_model.add_pipe("abbreviation_detector")
    return ann_model

def annotate(proc_file, model):
    '''
    :param proc_file: file to annotate
    :param model: spacy model to perform entity linking
    :return: annotated file with token level annotations of entities and sentence level annotations of evidence
    '''
    proc_file = pd.read_csv(proc_file, low_memory=False)
    linker = model.get_pipe("scispacy_linker")
    docs = model.pipe(proc_file["CLEANED_TEXT"].tolist())
    dataset_ann = []
    with open('umls_anns_.json', 'w') as umls_writer:
        for doc in docs:
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
        json.dump(dataset_ann, umls_writer)

    print(dataset_ann)
    print("Len of dataset {}".format(len(dataset_ann)))


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
