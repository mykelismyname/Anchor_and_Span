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
from scispacy.umls_linking import UmlsEntityLinker
from scispacy.umls_utils import UmlsKnowledgeBase
from glob import glob
from copy import deepcopy
from time import time
from pyspark.sql import SparkSession
from multiprocessing import Pool
import functools
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# spaCy isn't serializable but loading it is semi-expensive, therefore load the model only when its time to use it
SPACY_MODEL = None
def get_spacy_model():
    global SPACY_MODEL, linker
    if not SPACY_MODEL:
        SPACY_MODEL = spacy.load("en_core_sci_sm")
        linker = EntityLinker(linker_name="umls")
    return SPACY_MODEL, linker

def annotate_entities_linker_unembedded(document):
    '''
    :param document: text to annotate
    :return: annotated file with token level annotations of entities and sentence level annotations of evidence
    '''
    model, linker = get_spacy_model()
    dataset_ann = []

    doc_ann = {}
    sentences = []
    sent_entities = []
    doc = model(document)

    for sent_id,sent in enumerate(doc.sents):
        for entity in sent.ents:
            entity_annotation = {"name": "{}".format(entity),
                                 "pos": [entity.start, entity.end],
                                 "sent_id": sent_id,
                                 "linked_umls_entities": []}
            try:
                linked_entity = {}
                cuis = linker.kb.alias_to_cuis[entity.text]
                linked_entities = []
                for cui in cuis:
                    umls_ent = linker.kb.cui_to_entity[cui]
                    linked_entity["cui"] = cui
                    for code in umls_ent.types:
                        styname = linker.kb.semantic_type_tree.get_canonical_name(code)
                        linked_entity["type_id"] = code
                        linked_entity["type"] = styname
                        entity_annotation["linked_umls_entities"].append(linked_entity)
            except KeyError:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                logging.warning("Entity - {} can't be found in UMLS".format(entity))

            sent_entities.append(entity_annotation)
        sentences.append([m.text for m in sent])
    doc_ann["Entities"] = sent_entities
    doc_ann["Sents"] = sentences
    dataset_ann.append(doc_ann)
    logging.info("Number of documents annotated {}".format(len(dataset_ann)))
    return dataset_ann

#annotating relations by searching left and right of the detected drugs and or diseases
def retrieve_annotations(data, write_to_disk=False):
    if os.path.isdir(data):
        files = glob(data+'/*.pkl')
        for file in files:
            annotated_text, annotated_df = annotate_relations(file)
            if write_to_disk:
                file_name, file_ext = data.split('.')
                dest_dir = os.path.dirname(data) + '/' + os.path.basename(data) + '_rels'
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)
                    wf = open(os.path.join(dest_dir, file_ext + "_rels.pkl"), 'wb')
                    pickle.dump(annotated_text, wf, protocol=pickle.HIGHEST_PROTOCOL)
    elif os.path.isfile(data):
        annotated_text, annotated_df = annotate_relations(data)
        if write_to_disk:
            file_name, file_ext = os.path.basename(data).split('.')
            dest_dir = os.path.dirname(data) + '/' + 'rel_anns'
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            wf = open(os.path.join(dest_dir, file_name+"_rels.pkl"), 'wb')
            pickle.dump(annotated_text, wf, protocol=pickle.HIGHEST_PROTOCOL)
        return annotated_df

def spacyProcess(data, model, dask=False):
    if dask:
        data.loc[:,"PROCESSED_TEXT"] = data['CLEANED_TEXT'].apply(lambda x:model(x))
        return data
    documents = model.pipe(data)
    return documents

def annotate_entities_linker_embedded(document, model):
    '''
    :param document: text to annotate
    :return: annotated file with token level annotations of entities and sentence level annotations of evidence
    '''
    linker = model.get_pipe("scispacy_linker")
    dataset_ann = []
    doc_ann = {}
    sentences = []
    sent_entities = []

    for sent_id, sent in enumerate(document.sents):
        for entity in sent.ents:
            entity_annotation = {"name": "{}".format(entity),
                                 "pos": [entity.start, entity.end],
                                 "sent_id": sent_id,
                                 "linked_umls_entities": []}
            try:
                linked_entities = []
                for e in entity._.kb_ents:
                    linked_entity = {}
                    cui, cui_score = e
                    umls_ent = linker.kb.cui_to_entity[cui]
                    linked_entity["cui"] = cui
                    linked_entity["score"] = np.round(cui_score, 2)
                    for code in umls_ent.types:
                        styname = linker.kb.semantic_type_tree.get_canonical_name(code)
                        linked_entity["type_id"] = code
                        linked_entity["type"] = styname
                        entity_annotation["linked_umls_entities"].append(linked_entity)
            except KeyError:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                logging.warning("Entity - {} can't be found in UMLS".format(entity))
            sent_entities.append(entity_annotation)
        sentences.append([m.text for m in sent])
    doc_ann["Entities"] = sent_entities
    doc_ann["Sents"] = sentences
    dataset_ann.append(doc_ann)
    return dataset_ann


#annotating relations by searching left and right of the detected drugs and or diseases
def annotate_relations(file):
    file_annotated = []
    file_df_list, file_df = [], pd.DataFrame()
    sentences_ann = {}

    with open(file, 'rb') as rf:
        doc_annotations = pickle.load(rf)
        for i, ann in enumerate(doc_annotations):
            labels, labels_found = [], False
            entity_annotations = ann['Entities']
            sentences_ann['Entities'] = deepcopy(entity_annotations)
            sentences_ann['Sents'] = ann['Sents']
            drugs, diseases = [], []
            # logging.info("Document %d\n"%(i))
            for e in range(len(entity_annotations)):
                # print(e,":",entity_annotations[e][0])
                linked_entity_types = entity_annotations[e][0][
                    'linked_umls_entities']  # select linked entity with highest score
                if len(linked_entity_types) > 0:
                    if linked_entity_types[0]['type'].lower() == 'clinical drug':
                        drugs.append(entity_annotations[e][0]['name'])
                        # print("Drug:", entity_annotations[e][0]['name'], entity_annotations[e][0]['sent_id'])
                        # search to the left and to the right
                        l_lab, r_lab = search_linked_entities_left_and_right(e, entity_annotations, ann['Sents'],
                                                                             'drug_sign or symptom')
                        unpacked_labels = unpack_relation_and_evidence_labels([r_lab, l_lab])
                        labels.extend(unpacked_labels)

                    if linked_entity_types[0]['type'].lower() == 'disease or syndrome':
                        diseases.append(entity_annotations[e][0]['name'])
                        # print("Disease:%s"%(e), entity_annotations[e][0]['name'], entity_annotations[e][0]['sent_id'])
                        # search to the left and to the right
                        l_lab, r_lab = search_linked_entities_left_and_right(e, entity_annotations, ann['Sents'],
                                                                             'disease_sign or symptom')
                        unpacked_labels = unpack_relation_and_evidence_labels([r_lab, l_lab])
                        labels.extend(unpacked_labels)

            sentences_ann['labels'] = labels
            file_annotated.append(sentences_ann)
            if labels:
                disease_symptom = [i for i in labels if i['rel'] == 'disease_symptom_relation']
                drug_symptom = [i for i in labels if i['rel'] == 'drug_symptom_relation']
                sents = [" ".join(y) for x, y in enumerate(ann['Sents'])]
                sents_joined = " ".join(i for i in sents)
                df = pd.DataFrame([sents_joined], columns=['Sentences'])
                if disease_symptom:
                    disease_symptom_df = pd.DataFrame(disease_symptom)
                    disease_symptom_df['h'] = disease_symptom_df['h'].apply(lambda x: entity_annotations[int(x)][0]['name'])
                    disease_symptom_df['t'] = disease_symptom_df['t'].apply(lambda x: entity_annotations[int(x)][0]['name'])
                    disease_symptom_df.columns = ["Relation_1", "Disease", "Dis-Sym", "Evid-Dis-Sym"]
                    df = pd.concat([df, disease_symptom_df], axis=1)
                    un_matched_diseases = [i for i in diseases if i not in df['Disease'].tolist()]
                    if un_matched_diseases:
                        print("Unmatched Diseases:", un_matched_diseases)
                if drug_symptom:
                    drug_symptom_df = pd.DataFrame(drug_symptom)
                    drug_symptom_df['h'] = drug_symptom_df['h'].apply(lambda x: entity_annotations[int(x)][0]['name'])
                    drug_symptom_df['t'] = drug_symptom_df['t'].apply(lambda x: entity_annotations[int(x)][0]['name'])
                    drug_symptom_df.columns = ["Relation_2", "Drug", "Dru-Sym", "Evid-Dru-Sym"]
                    df = pd.concat([df, drug_symptom_df], axis=1)
                    un_matched_drugs = [i for i in drugs if i not in drug_symptom_df['Drug'].tolist()]
                    if un_matched_drugs:
                        print("Unmatched drugs:", un_matched_drugs)

                df = df.reset_index(drop=True)
                file_df = pd.concat([file_df, df], ignore_index=True, axis=0)

    return file_annotated, file_df

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
    st = time()
    data = pd.read_csv(args.data, low_memory=False)
    annotation_size = args.annotation_size.split()
    if len(annotation_size) > 1:
        ann_window = (int(annotation_size[0]), int(annotation_size[1]))
        start, end = ann_window
        data = data[start:end]

    if 'entities' in annotate_type:
        if args.model_linker_embedded:
            SPACY_MODEL = spacy.load("en_core_sci_sm")
            SPACY_MODEL.add_pipe("abbreviation_detector")
            SPACY_MODEL.add_pipe("scispacy_linker", config={"linker_name": "umls"})
            documents = spacyProcess(SPACY_MODEL, data["CLEANED_TEXT"])
            with Pool() as pool:
                _pt_ = functools.partial(annotate_entities_linker_unembedded, SPACY_MODEL)
                dataset_annotated = pool.map(_pt_, documents)
                _dataset_annotated = [i[0] for i in dataset_annotated.collect()]
        else:
            data_read_spark = spark.createDataFrame(data)
            data_spark = data_read_spark.select("CLEANED_TEXT").rdd.flatMap(lambda x: x)

            dataset_annotated = data_spark.map(lambda x: annotate_entities_linker_unembedded(x))
            _dataset_annotated = [i[0] for i in dataset_annotated.collect()]

        with open(args.dest + '/umls_anns_.json', 'w') as umls_writer:
            json.dump(_dataset_annotated, umls_writer)

        logging.info("Total time taken {}s".format(time() - st))
    if 'relations' in annotate_type:
        #python scisp_ann.py - -data ../anns/spacy/ --annotate_type 'relations'
        annotate_relations(args.data)
    if 'retrieve_annotations' in annotate_type:
        retrieve_annotations(args.data)

if __name__ == '__main__':
    par = argparse.ArgumentParser()
    par.add_argument('--data', default='../NOTESEVENTS_CLEANED.csv', help='location of the data')
    par.add_argument('--dest', default='../anns/spacy', help='location of the data')
    par.add_argument('--model', default='en_core_sci_sm', help='scispacy biomedical model')
    par.add_argument('--annotate_type', default='entities', help="what to annotate 'entities', or 'relations' or 'entities relations'")
    par.add_argument('--kg', type=str, default='umls', help='which knoiwledge graph is linked to for annotation')
    par.add_argument('--annotation_size', type=str, default="-1", help='number of documents to annotate')
    par.add_argument('--model_linker_embedded', action="store_true",
                     help='linker can either be embedded in model or introduced after detecting entities')

    args = par.parse_args()
    spark = SparkSession.builder.master("local").appName("Mimic-II").getOrCreate()
    # UmlsKnowledgeBase()
    main(args)