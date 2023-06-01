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

class annotate:
    def __init__(self, file, model):
        if os.path.basename(file).endswith('csv.gz'):
            self.file_df = pd.read_csv(file, compression="gzip", low_memory=False)
        else:
            self.file_df = pd.read_csv(file, low_memory=False)
        self.model = model

    def update_concepts(self, concepts):
        type_ids_filter = concepts
        cui_filters = set()
        for type_ids in type_ids_filter:
            cui_filters.update(self.model.cdb.addl_info['type_id2cuis'][type_ids])
        self.model.cdb.config.linking['filters']['cuis'] = cui_filters
        print("The size of the cdb is now: %s"%(len(cui_filters)))
        return self.model

def clean(s):
    s = re.sub(r"[\n]+", " ", s)
    s = re.sub(r"(\[\*\*)|(\*\*\])", "", s)
    s = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", s)
    s = re.sub(r'(Admission Date:\s*\d{4}\-\d{1,2}\-\d{1,2} Discharge Date:\s*\d{4}\-\d{1,2}\-\d{1,2})', '', s)
    s = re.sub(r"\s+", " ", s)
    return s

# format the df to match: required input data for multiprocessing = [(doc_id, doc_text), (doc_id, doc_text), ...]
def data_iterator(data):
    for id, row in data[['CLEANED_TEXT']].iterrows():
        print(row)
        yield (id, str(row['CLEANED_TEXT']))

def main(args):
    model_pack_path = args.model

    model_path = CAT.load_model_pack(model_pack_path)
    ann = annotate(file=args.data, model=model_path)
    ann_file = ann.file_df
    if args.cleaning:
        print('------------------------DATASET PREPARATION------------------------')
        ann_file['CLEANED_TEXT'] = ann_file['TEXT'].apply(lambda k:clean(k))
        ann_file.to_csv('../NOTESEVENTS_CLEANED.csv')
    ann_model = ann.update_concepts(args.concepts)
    print("Number of entries %s"%(int(len(ann_file))))

    ann_window = args.annotation_size.split()
    start, end = int(ann_window[0]), int(ann_window[1])

    if end > -1:
        ann_file_ = ann_file[start:end]
    else:
        ann_file_ = ann_file.copy()
    print('------------------------ANNOTATION BEGINS------------------------')
    st_time = time.time()
    if not args.multiprocessing:
        ann_file_.loc[:,'ENTITIES'] = ann_file_.loc[:,'CLEANED_TEXT'].apply(lambda x:ann_model.get_entities(x))
        annotations = ann_file_['ENTITIES'].tolist()
    else:
        print('------------------------multi processing------------------------')
        annotations = ann_model.multiprocessing(data_iterator(ann_file_),
                                      batch_size_chars=args.batch_size_chars,
                                      nproc=8)

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)
    print('------------------------ANNOTATION ENDS------------------------')
    et_time = time.time()
    print('Total time taken - {}mins'.format((et_time - st_time)/60))
    annotation_file = 'anns_mult_{}_{}.pkl'.format(start, end) if args.multiprocessing else 'anns_{}_{}.pkl'.format(start, end)
    with open(os.path.join(args.dest, annotation_file), 'wb') as a:
        pickle.dump(annotations, a, protocol=pickle.HIGHEST_PROTOCOL)
        a.close()

if __name__ == '__main__':
    par = ArgumentParser()
    par.add_argument('--data', default='../NOTEEVENTS.csv.gz', help='data')
    par.add_argument('--dest', default='../anns/', help='location to store annotation file')
    par.add_argument('--model', default='models/medmen_wstatus_2021_oct.zip', type=str, help='model location')
    par.add_argument('--cleaning', action='store_true', help='clean the dataset or not')
    par.add_argument('--annotation_size', type=str, help='number of documents to annotate')
    par.add_argument('--concepts', default=['T047', 'T048', 'T200', 'T184'], type=list, help='concepts to annotate')
    par.add_argument('--multiprocessing', default=False, type=bool, help='multiprocessing')
    par.add_argument('--batch_size_chars', default=500000, type=int, help='Batch size (BS) in number of characters')
    args = par.parse_args()
    main(args)
    


