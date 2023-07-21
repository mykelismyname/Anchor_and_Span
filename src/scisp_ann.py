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
from spacy.tokens import Doc, DocBin
from spacy.language import Language
from glob import glob
from copy import deepcopy
from time import time
from pyspark.sql import SparkSession
from dask.distributed import Client
import dask.dataframe as dd
import dask.array as da
import functools
import utils
pd.options.mode.chained_assignment = None
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# spaCy isn't serializable but loading it is semi-expensive, therefore load the model only when its time to use it
SPACY_MODEL = None
def get_spacy_model():
    global SPACY_MODEL, linker
    if not SPACY_MODEL:
        SPACY_MODEL = spacy.load(_model_)
        linker = EntityLinker(linker_name="umls")
    return SPACY_MODEL, linker

#using spacy multiprocess pip function to multiprocess data
def spacyProcess(data, model, dask=False):
    if dask:
        data.loc[:,"PROCESSED_TEXT"] = data['CLEANED_TEXT'].apply(lambda x:model(x))
        return data
    # model.add_pipe("annotate", name="retrieve_annotate", last=True)
    documents = []
    for d in model.pipe(data):
        documents.append(d._.umls_ents)
    return documents

#function to call the actual model embedded annotation function
def call_annotate_emb(data):
    data.loc[:, "ANNOTATED_TEXT"] = data["PROCESSED_TEXT"].apply(
        lambda x: annotate_entities_linker_embedded(x))
    return data

#function to call the actual model unembedded annotation function
def call_annotate_unemb(data, multi_processor=False):
    data.loc[:, "ANNOTATED_TEXT"] = data["CLEANED_TEXT"].apply(
        lambda x: annotate_entities_linker_unembedded(document=x, multi_processor=multi_processor))
    return data

#model is used to detect entities and linker introduced to link detected entities
def annotate_entities_linker_unembedded(document, multi_processor=False):
    '''
    :param document: text to annotate
    :return: annotated file with token level annotations of entities and sentence level annotations of evidence
    '''
    model, linker = get_spacy_model()
    if multi_processor:
        doc = model(document)
    else:
        doc = document

    doc_ann = {}
    sentences = []
    sent_entities = []

    for sent_id,sent in enumerate(doc.sents):
        for entity in sent.ents:
            entity_annotation = {"name": "{}".format(entity),
                                 "pos": [entity.start, entity.end],
                                 "sent_id": sent_id}
            try:
                linked_entity = {}
                cuis = linker.kb.alias_to_cuis[entity.text]
                linked_codes = []
                for cui in cuis:
                    umls_ent = linker.kb.cui_to_entity[cui]
                    linked_entity[cui] = {}
                    for code in umls_ent.types:
                        if code not in linked_codes:
                            styname = linker.kb.semantic_type_tree.get_canonical_name(code)
                            linked_entity[cui]["type_id"] = code
                            linked_entity[cui]["type"] = styname
                            linked_codes.append(code)
                        else:
                            pass
                entity_annotation["linked_umls_entities"] = linked_entity
            except KeyError:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                entity_annotation["linked_umls_entities"] = {}
                logging.warning("Entity - {} can't be found in UMLS".format(entity))

            sent_entities.append(entity_annotation)
        sentences.append([m.text for m in sent])
    doc_ann["Entities"] = sent_entities
    doc_ann["Sents"] = sentences
    return doc_ann

#model is used to detect entities and linker introduced to link detected entities
@Language.component("annotate")
def annotate_entities_linker_embedded(document):
    '''
    :param document: text to annotate
    :return: annotated file with token level annotations of entities and sentence level annotations of evidence
    '''
    doc_ann = {}
    sentences = []
    sent_entities = []

    for sent_id, sent in enumerate(document.sents):
        for entity in sent.ents:
            entity_annotation = {"name": "{}".format(entity),
                                 "pos": [entity.start, entity.end],
                                 "sent_id": sent_id}
            try:
                linked_codes = []
                linked_entity = {}
                for e in entity._.kb_ents:
                    cui, cui_score = e
                    umls_ent = linker.kb.cui_to_entity[cui]
                    linked_entity[cui] = {}
                    linked_entity[cui]["score"] = np.round(cui_score, 2)
                    for code in umls_ent.types:
                        if code not in linked_codes:
                            styname = linker.kb.semantic_type_tree.get_canonical_name(code)
                            linked_entity[cui]["type_id"] = code
                            linked_entity[cui]["type"] = styname
                            linked_codes.append(code)
                        else:
                            pass
                entity_annotation["linked_umls_entities"] = linked_entity
            except KeyError:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                entity_annotation["linked_umls_entities"] = {}
                logging.warning("Entity - {} can't be found in UMLS".format(entity))
            sent_entities.append(entity_annotation)
        sentences.append([m.text for m in sent])
    doc_ann["Entities"] = sent_entities
    doc_ann["Sents"] = sentences
    return doc_ann

def main(args):
    st = time()
    if args.dask or args.spark:
        data_set = dd.read_csv(args.data, encoding="utf-8", engine="python", on_bad_lines="warn", dtype={"CLEANED_TEXT": str})
        data_ = data_set[["CLEANED_TEXT"]]
        logging.info("There are {} partitions in the dataset".format(data_.npartitions))
    else:
        # chunks = pd.read_csv(args.data, chunksize=1000000, low_memory=False)
        # data_set = pd.concat(chunks)
        data_set = dd.read_csv(args.data, encoding="utf-8", engine="python", on_bad_lines="warn", dtype={"CLEANED_TEXT": str})
        data_ = data_set[["CLEANED_TEXT"]].compute()

    annotation_size = args.annotation_size.split()
    if len(annotation_size) > 1:
        ann_window = (int(annotation_size[0]), int(annotation_size[1]))
        start, end = ann_window
    else:
        start, end = 0, len(data_)

    logging.info("Dataset successfully loaded")

    if args.model_linker_embedded:
        if args.dask:
            data = data_.loc[start:end]
            documents = data.map_partitions(spacyProcess, model=SP_MODEL, dask=True, meta=pd.DataFrame(columns=["CLEANED_TEXT", "PROCESSED_TEXT"]))
            logging.info("Completed Spacy processing")
            dataset_annotated = documents[["PROCESSED_TEXT"]].map_partitions(call_annotate_emb, meta=pd.DataFrame(columns=["PROCESSED_TEXT", "ANNOTATED_TEXT"]))
            logging.info("Completed annotation")
            logging.info("Number of partitions at this stage {}".format(dataset_annotated.npartitions))
            _dataset_annotated = [i for i in dataset_annotated["ANNOTATED_TEXT"].compute()]
            print(_dataset_annotated)
        elif args.spark:
            #process spark dataframe
            pass
        else:
            data_ = data_[start:end]
            cleaned_text = data_["CLEANED_TEXT"].tolist()
            documents = spacyProcess(cleaned_text, SP_MODEL)
            logging.info("Completed Spacy processing")
            _dataset_annotated = documents
            logging.info("Completed annotation")
    else:
        if args.dask:
            data = data_.loc[start:end]
            dataset_annotated = data.map_partitions(call_annotate_unemb, multi_processor=True, meta=pd.DataFrame(columns=["CLEANED_TEXT", "ANNOTATED_TEXT"]))
            logging.info("Completed annotation")
            _dataset_annotated = dataset_annotated["ANNOTATED_TEXT"].compute()
        elif args.spark:
            _data_ = data_.compute()
            data_read_spark = spark.createDataFrame(_data_)
            data_spark = data_read_spark.select("CLEANED_TEXT").rdd.flatMap(lambda x: x)
            dataset_annotated = data_spark.map(lambda x:annotate_entities_linker_unembedded(x, multi_processor=True))
            logging.info("Completed annotation")
            _dataset_annotated = dataset_annotated.collect()
        else:
            SPACY_MODEL = spacy.load(_model_)
            cleaned_text = data_["CLEANED_TEXT"].tolist()
            documents = spacyProcess(cleaned_text, SPACY_MODEL)
            logging.info("Completed Spacy processing")
            _dataset_annotated = [annotate_entities_linker_unembedded(i) for i in documents]
            logging.info("Completed annotation")
            #_dataset_annotated = [i[0] for i in dataset_annotated]

    dest = utils.createDir(os.path.abspath(args.dest))
    processor = 'dask' if args.dask else 'spark' if args.spark else 'spacy'
    if args.pickle_output:
        dest_file = os.path.join(dest, 'anns_{}_{}_{}.pkl'.format(str(start), str(end), processor))
        with open(dest_file, 'wb') as wf:
            pickle.dump(_dataset_annotated, wf, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        dest_file = os.path.join(dest, 'anns_{}_{}_{}.json'.format(str(start), str(end), processor))
        with open(dest_file, 'w') as wf:
            json.dump(_dataset_annotated, wf)
    print(args)
    logging.info("Number of documents annotated {}".format(len(_dataset_annotated)))
    logging.info("Total time taken {}s".format(time() - st))


if __name__ == '__main__':
    par = argparse.ArgumentParser()
    par.add_argument('--data', default='../NOTESEVENTS_CLEANED.csv', help='location of the data')
    par.add_argument('--dest', default='../anns/spacy', help='location of the data')
    par.add_argument('--annotation_size', type=str, default="-1", help='number of documents to annotate')
    par.add_argument('--model', default='en_core_sci_sm', help='scispacy biomedical model')
    par.add_argument('--kg', type=str, default='umls', help='which knoiwledge graph is linked to for annotation')
    par.add_argument('--dask', action="store_true", help='use dask for multi-processing')
    par.add_argument('--spark', action="store_true", help='use spark for multi-processing')
    par.add_argument('--model_linker_embedded', action="store_true", help='linker can either be embedded in model or introduced after detecting entities')
    par.add_argument('--pickle_output', action="store_true", help='save output as a pickle object')
    args = par.parse_args()
    # client = Client()

    _model_ = args.model
    # used if args.model_linker_embedded is et to True
    SP_MODEL = spacy.load(args.model)
    SP_MODEL.add_pipe("scispacy_linker", config={"linker_name": "umls"})
    Doc.set_extension("umls_ents", getter=annotate_entities_linker_embedded)
    linker = SP_MODEL.get_pipe("scispacy_linker")

    spark = SparkSession.builder.master("local").appName("Mimic-II").config("spark.driver.memory", "15g").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    main(args)