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

#annotating relations by searching left and right of the detected drugs and or diseases
def annotate_relations(data):
    files = glob(data+'/*.pkl')
    for file in files:
        with open(file, 'rb') as rf, open("test.pkl", 'wb') as wf:
            doc_annotations = pickle.load(rf)
            file_annotated = []
            sentences_ann = {}
            for i,ann in enumerate(doc_annotations):
                labels, labels_found = [], False
                entity_annotations = ann['Entities']
                sentences_ann['Entities'] = deepcopy(entity_annotations)
                sentences_ann['Sents'] = ann['Sents']
                drugs, diseases = [], []
                # logging.info("Document %d\n"%(i))
                for e in range(len(entity_annotations)):
                    # print(e,":",entity_annotations[e][0])
                    linked_entity_types = entity_annotations[e][0]['linked_umls_entities'] #select linked entity with highest score
                    if len(linked_entity_types) > 0:
                        if linked_entity_types[0]['type'].lower() == 'clinical drug':
                            drugs.append(entity_annotations[e][0]['name'])
                            # print("Drug:", entity_annotations[e][0]['name'], entity_annotations[e][0]['sent_id'])
                            #search to the left and to the right
                            l_lab, r_lab = search_linked_entities_left_and_right(e, entity_annotations, ann['Sents'], 'drug_sign or symptom')
                            unpacked_labels = unpack_relation_and_evidence_labels([r_lab, l_lab])
                            labels.extend(unpacked_labels)

                        if linked_entity_types[0]['type'].lower() == 'disease or syndrome':
                            diseases.append(entity_annotations[e][0]['name'])
                            # print("Disease:%s"%(e), entity_annotations[e][0]['name'], entity_annotations[e][0]['sent_id'])
                            # search to the left and to the right
                            l_lab, r_lab = search_linked_entities_left_and_right(e, entity_annotations, ann['Sents'], 'disease_sign or symptom')
                            unpacked_labels = unpack_relation_and_evidence_labels([r_lab, l_lab])
                            labels.extend(unpacked_labels)

                sentences_ann['labels'] = labels
                file_annotated.append(sentences_ann)
                if labels:
                    # for i,j in enumerate(entity_annotations):
                    #     print(i,":",j[0])
                    disease_symptom = [i for i in labels if i['rel'] == 'disease_symptom_relation']
                    drug_symptom = [i for i in labels if i['rel'] == 'drug_symptom_relation']
                    sents = [" ".join(y) for x,y in enumerate(ann['Sents'])]
                    sents_joined = " ".join(i for i in sents)
                    df = pd.DataFrame([sents_joined], columns=['Sentences'])
                    if disease_symptom:
                        disease_symptom_df = pd.DataFrame(disease_symptom).rename(columns={"h":"Disease", "t":"Symptom"})
                        disease_symptom_df['Disease'] = disease_symptom_df['Disease'].apply(lambda x: entity_annotations[int(x)][0]['name'])
                        disease_symptom_df['Symptom'] = disease_symptom_df['Symptom'].apply(lambda x: entity_annotations[int(x)][0]['name'])
                        df = pd.concat([df, disease_symptom_df], axis=1)
                    if drug_symptom:
                        drug_symptom_df = pd.DataFrame(drug_symptom).rename(columns={"h":"Drug", "t":"Symptom"})
                        drug_symptom_df['Drug'] = drug_symptom_df['Drug'].apply(lambda x: entity_annotations[int(x)][0]['name'])
                        drug_symptom_df['Symptom'] = drug_symptom_df['Symptom'].apply(lambda x: entity_annotations[int(x)][0]['name'])
                        df = pd.concat([df, drug_symptom_df], axis=1)

                    print(df)

            # pickle.dump(file_annotated, wf, protocol=pickle.HIGHEST_PROTOCOL)

def search_linked_entities_left_and_right(e, entity_annotations, sentences, umls_rel_type, search_window=-1):
    l, r = e - 1, e + 1
    e1_entity = entity_annotations[e][0]['name']
    l_lab, r_lab = {e: {}}, {e: {}}
    search_so_far_right, search_so_far_left = 0, 0
    umls_type, umls_rel = umls_rel_type.split('_')
    while l >= 0 or r < len(entity_annotations):
        left_related_entities = entity_annotations[l][0]['linked_umls_entities'] if l >= 0 else []
        right_related_entities = entity_annotations[r][0]['linked_umls_entities'] if r < len(entity_annotations) else []

        if len(left_related_entities) > 0:
            e2_entity = entity_annotations[l][0]['name']
            # print('left:', l, entity_annotations[l][0]['name'])
            if left_related_entities[0]['type'].lower() ==  umls_rel:
                # print(entity_annotations[e][0]['name'], entity_annotations[l][0]['name'])
                evidence_sentence = entity_annotations[l][0]['sent_id']
                l_ent = str(l) + '_' + umls_type + '_symptom_relation'
                if l_ent not in l_lab[e]:
                    l_lab[e][l_ent] = [evidence_sentence]
                else:
                    if evidence_sentence not in l_lab[e][l_ent]:
                        l_lab[e][l_ent].append(evidence_sentence)

                search_so_far_left += len(sentences[evidence_sentence])
                if search_so_far_left == search_window:
                    break

        if len(right_related_entities) > 0:
            e2_entity = entity_annotations[r][0]['name']
            # print('right:', r, entity_annotations[r][0]['name'])
            if right_related_entities[0]['type'].lower() ==  umls_rel:
                # print(entity_annotations[e][0]['name'], entity_annotations[r][0]['name'])
                evidence_sentence = entity_annotations[r][0]['sent_id']
                r_ent = str(r) + '_' + umls_type + '_symptom_relation'
                if r_ent not in r_lab[e]:
                    r_lab[e][r_ent] = [evidence_sentence]
                else:
                    if evidence_sentence not in r_lab[e][r_ent]:
                        r_lab[e][r_ent].append(evidence_sentence)

                search_so_far_right += len(sentences[evidence_sentence])
                if search_so_far_right == search_window:
                    break
        l -= 1
        r += 1
    return l_lab, r_lab

#unpacking the relation and evidence labels one by one from a dict of dicts
def unpack_relation_and_evidence_labels(labels):
    unpacked_labels = []
    for x in labels:
        if len(x) > 0:
            for m in x:
                if len(x[m]) > 0:
                    for y in x[m]:
                        x_r = {}
                        y_str, relation = y.split('_', 1)
                        x_r["rel"] = relation
                        x_r['h'] = m
                        x_r['t'] = y_str
                        x_r['evidence'] = x[m][y]
                        unpacked_labels.append(x_r)
    return unpacked_labels

def main(args):
    annotate_type = args.annotate_type.split()
    if 'entities' in annotate_type:
        ann_model = load_model(args.model)
        ann_model.add_pipe("scispacy_linker", config={"linker_name": "umls"})
        #python scisp_ann.py --data ../NOTESEVENTS_CLEANED.csv --annotation_size "20000 22000"
        annotate_entities(proc_file=args.data, model=ann_model, dest=args.dest)
    if 'relations' in annotate_type:
        annotate_relations(args.data)
        # with open(args.data, 'rb') as rd:
        #     d = pickle.load(rd)
        #     for i in d:
        #         print(i)
        #         break

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