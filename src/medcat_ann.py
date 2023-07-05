import pandas as pd
import os
from glob import glob
from medcat.cat import CAT
from medcat.cdb import CDB
from medcat.config import Config
from medcat.vocab import Vocab
from medcat.meta_cat import MetaCAT
from medcat.preprocessing.tokenizers import TokenizerWrapperBPE
from tokenizers import ByteLevelBPETokenizer
from argparse import ArgumentParser
import json
import re
import time
import pickle
import logging

pd.options.mode.chained_assignment = None
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# format the df to match: required input data for multiprocessing = [(doc_id, doc_text), (doc_id, doc_text), ...]
def data_iterator(data):
    for id, row in data[['CLEANED_TEXT']].iterrows():
        yield (id, str(row['CLEANED_TEXT']))

#update model voabularly with concept types you might be interested
def update_concepts(model, concepts):
    type_ids_filter = concepts
    cui_filters = set()
    for type_ids in type_ids_filter:
        cui_filters.update(self.model.cdb.addl_info['type_id2cuis'][type_ids])
    model.cdb.config.linking['filters']['cuis'] = cui_filters
    print("The size of the cdb is now: %s"%(len(cui_filters)))
    return model

def main(args):
    st_time = time.time()
    #load dataset
    data_set = dd.read_csv(args.data, encoding="utf-8", engine="python", on_bad_lines="warn")
    data_ = data_set[["CLEANED_TEXT"]]

    annotation_size = args.annotation_size.split()
    if len(annotation_size) > 1:
        ann_window = (int(annotation_size[0]), int(annotation_size[1]))
        start, end = ann_window
        _data_ = data_.compute()[start:end]
    else:
        _data_ = data_.compute()
        start, end = 0, len(_data_)

    #load MEDCAT model
    medcat_model = CAT.load_model_pack(args.medcat_model)

    if args.update_concepts:
        concepts = args.concepts
        model = update_concepts(medcat_model, concepts)

    logging.info('---ANNOTATION BEGINS---')
    st_time = time.time()
    if not args.multiprocessing:
        ann_file_.loc[:,'ENTITIES'] = ann_file_.loc[:,'CLEANED_TEXT'].apply(lambda x:ann_model.get_entities(x))
        annotations = ann_file_['ENTITIES'].tolist()
    else:
        logging.info('---multi processing---')
        annotations = ann_model.multiprocessing(data_iterator(ann_file_),
                                      batch_size_chars=args.batch_size_chars,
                                      nproc=8)

    dest = utils.createDir(args.dest)
    print('------------------------ANNOTATION ENDS------------------------')
    et_time = time.time()

    annotation_file = 'anns_mult_{}_{}.pkl'.format(start, end) if args.multiprocessing else 'anns_{}_{}.pkl'.format(start, end)
    with open(os.path.join(args.dest, annotation_file), 'wb') as a:
        pickle.dump(annotations, a, protocol=pickle.HIGHEST_PROTOCOL)
        a.close()

    logging.info("Total time taken {}s".format(time.time() - st))

if __name__ == '__main__':
    par = ArgumentParser()
    par.add_argument('--data', default='../NOTEEVENTS.csv.gz', help='data')
    par.add_argument('--dest', default='../anns/', help='location to store annotation file')
    par.add_argument('--medcat_model', default='../models/medmen_wstatus_2021_oct.zip', type=str, help='model location')
    par.add_argument('--cleaning', action='store_true', help='clean the dataset or not')
    par.add_argument('--annotation_size', type=str, help='number of documents to annotate')
    par.add_argument('--update_concepts', action='store_true', help='update model concepts')
    par.add_argument('--concepts', default=['T047', 'T048', 'T200', 'T184'], type=list, help='concepts to annotate')
    par.add_argument('--multiprocessing', default=False, type=bool, help='multiprocessing')
    par.add_argument('--batch_size_chars', default=500000, type=int, help='Batch size (BS) in number of characters')
    args = par.parse_args()
    main(args)



