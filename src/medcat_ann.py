import pandas as pd
import os
from glob import glob
from medcat.cat import CAT
from argparse import ArgumentParser
import json
import re
import time
import pickle
import logging
import spacy
import dask.dataframe as dd
import utils

pd.options.mode.chained_assignment = None
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# format the df to match: required input data for multiprocessing = [(doc_id, doc_text), (doc_id, doc_text), ...]
def data_iterator(data):
    for id, row in data[['PROCESSED_TEXT']].iterrows():
        yield (id, str(row['PROCESSED_TEXT']))

#update model voabularly with concept types you might be interested
def update_concepts(model, concepts):
    type_ids_filter = concepts
    cui_filters = set()
    for type_ids in type_ids_filter:
        cui_filters.update(model.cdb.addl_info['type_id2cuis'][type_ids])
    model.cdb.config.linking['filters']['cuis'] = cui_filters
    print("The size of the cdb is now: %s"%(len(cui_filters)))
    return model

#using spacy multiprocess pip function to multiprocess data
def spacyProcess(data, dask=False):
    if dask:
        data.loc[:,"PROCESSED_TEXT"] = data['CLEANED_TEXT'].apply(lambda x:spacy_model(x))
        return data
    documents = spacy_model.pipe(data['CLEANED_TEXT'].tolist())
    return documents

#create a sentence tracker for each document
def createDocumentMap(documents):
    document_sent_map = {}
    sentences = []
    sent_id = 0
    for doc_id, doc in enumerate(documents):
        doc_sents = doc.sents
        doc_sents_ids = []
        for sent in doc_sents:
            sentences.append(sent.text)
            doc_sents_ids.append(sent_id)
            sent_id += 1
        document_sent_map[doc_id] = (doc_sents_ids[0], doc_sents_ids[-1] + 1)
    data_read = pd.DataFrame({'PROCESSED_TEXT': sentences})
    return data_read, sentences, document_sent_map

def extractAnnotations(results, sentences, document_sent_map):
    dataset_ann = []
    for doc_id, document_sent_range in document_sent_map.items():
        print(doc_id, document_sent_range)
        doc_ann = {}
        doc_entities = []
        sents = []
        for sent_id, doc in enumerate(results):
            if sent_id in range(document_sent_range[0], document_sent_range[-1]):
                sents.append(sentences[sent_id])
                print(sentences[sent_id])
                # print(results[doc]['entities'])
                for k, v in results[doc]['entities'].items():
                    entity = {}
                    _entity_ = (v['start'], v['end'], v['source_value'])
                    entity_text, entity_span_pos = utils.fetch_entitis_span_pos(_entity_, sentences[sent_id], "MEDCAT")
                    entity['name'] = v['source_value']
                    entity['sent_id'] = sent_id
                    print(entity_span_pos)
                    try:
                        assert len(entity_span_pos) == 2
                        entity['pos'] = [entity_span_pos[0], entity_span_pos[-1]]
                    except AssertionError:
                        entity['pos'] = []
                    entity['score'] = v['acc']
                    entity['linked_entities'] = []
                    linked_entity = {}
                    for m, n in zip(v['types'], v['type_ids']):
                        linked_entity['cui'] = v['cui']
                        linked_entity['type'] = m
                        linked_entity['type_id'] = n
                        entity['linked_entities'].append(linked_entity)
                    if entity:
                        print(entity)
                        doc_entities.append(entity)
            if sent_id == (document_sent_range[-1] - 1):
                print(
                    "=====================================================================================================")
                break
        doc_ann['Entities'] = doc_entities
        doc_ann['Sents'] = sents
        # doc_ann['Sents'] = [sentences[i] for i in range(document_sent_range[0], document_sent_range[-1])]
        # print(doc_ann)
        dataset_ann.append(doc_ann)
    return dataset_ann

def main(args):
    st_time = time.time()
    # load dataset
    data_set = dd.read_csv(args.data_dir, encoding="utf-8", engine="python")
    data_ = data_set[["CLEANED_TEXT"]]

    annotation_size = args.annotation_size.split()
    if len(annotation_size) > 1:
        ann_window = (int(annotation_size[0]), int(annotation_size[1]))
        start, end = ann_window
        _data_ = data_.compute()[start:end]
    else:
        _data_ = data_.compute()
        start, end = 0, len(_data_)

    logging.info("Dataset successfully loaded")

    # spacy process the data
    processed_data = spacyProcess(_data_)

    processed_data, sentences, document_sent_map = createDocumentMap(processed_data)

    # load MEDCAT model
    medcat_model = CAT.load_model_pack(args.medcat_model)

    if args.update_concepts:
        concepts = args.concepts
        medcat_model = update_concepts(medcat_model, concepts)

    logging.info('---ANNOTATION BEGINS---')
    st_time = time.time()
    if not args.multiprocessing:
        _data_.loc[:, 'ENTITIES'] = ann_file_.loc[:, 'CLEANED_TEXT'].apply(lambda x: medcat_model.get_entities(x))
        annotations = ann_file_['ENTITIES'].tolist()
    else:
        logging.info('---multi processing---')
        annotations = medcat_model.multiprocessing(data_iterator(processed_data),
                                                   batch_size_chars=args.batch_size_chars)

    dataset_annotated = extractAnnotations(annotations, sentences, document_sent_map)

    dest = utils.createDir(args.dest)
    print('------------------------ANNOTATION ENDS------------------------')

    # annotation_file = 'anns_mult_{}_{}.pkl'.format(start, end) if args.multiprocessing else 'anns_{}_{}.pkl'.format(start, end)
    # with open(os.path.join(args.dest, annotation_file), 'wb') as a:
    #     pickle.dump(annotations, a, protocol=pickle.HIGHEST_PROTOCOL)
    #     a.close()

    logging.info("Total time taken {}s".format(time.time() - st_time))

if __name__ == '__main__':
    par = ArgumentParser()
    par.add_argument('--data_dir', default='../NOTEEVENTS.csv.gz', help='data')
    par.add_argument('--dest', default='../anns/', help='location to store annotation file')
    par.add_argument('--medcat_model', default='../models/medmen_wstatus_2021_oct.zip', type=str, help='model location')
    par.add_argument('--spacy_model', default='en_core_sci_sm', help='spacy model for pre-processing')
    par.add_argument('--cleaning', action='store_true', help='clean the dataset or not')
    par.add_argument('--annotation_size', default="-1", type=str, help='number of documents to annotate')
    par.add_argument('--update_concepts', action='store_true', help='update model concepts')
    par.add_argument('--concepts', default=['T047', 'T048', 'T200', 'T184'], type=list, help='concepts to annotate')
    par.add_argument('--multiprocessing', action='store_true', help='multiprocessing')
    par.add_argument('--batch_size_chars', default=1000000, type=int, help='Batch size (BS) in number of characters')
    args = par.parse_args()
    # specifying spacy and scispacy model parameters
    spacy_model = spacy.load(args.spacy_model)
    main(args)



