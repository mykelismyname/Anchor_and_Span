import numpy as np
import pickle
import pandas as pd
import json
import logging
import argparse
import sys
import os
import yaml
import typing
import spacy
import scispacy
from scispacy.linking import EntityLinker
from scispacy.umls_utils import UmlsKnowledgeBase
from glob import glob
from copy import deepcopy
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient, HealthcareEntityRelation
import utils

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def ms_annotation(documents_segmented):
    errorsInAnnotation = {}
    dataset_ann = []
    for d, document in enumerate(documents_segmented):
        document_batches = utils.batching_for_textanalyticsclinet(25, len(document))
        doc_ann = {}
        document_batch_entities = []
        document_batch_sents = []
        # print("\n*****************************************************************************************************************************************************************************************************************************\n")
        sent_id = 0
        for k, document_batch in enumerate(document_batches):
            print(document_batch)
            start, end = document_batch
            document_sents = document[start:end]
            poller = text_analytics_client.begin_analyze_healthcare_entities(document_sents)
            result = poller.result()
            # docs = [doc for doc in result if not doc.is_error]
            errorSentences = []
            sent_entities = []
            for i, doc in enumerate(result):
                # print([(i,j) for i,j in enumerate(document_sents[i].split())])
                if not doc.is_error:
                    for entity in doc.entities:
                        entity_span_pos = utils.fecth_entitis_span_pos(entity, document_sents[i])
                        entity_annotation = {"name": "{}".format(entity.text),
                                             "pos": entity_span_pos,
                                             "sent_id": sent_id,
                                             "score": entity.confidence_score,
                                             "linked_entities": []}

                        if entity.data_sources is not None:
                            for data_source in entity.data_sources:
                                linked_entity = {}
                                if data_source.name.lower() == 'umls':
                                    try:
                                        umls_ent = kb.cui_to_entity[data_source.entity_id]
                                        stycodes = [(stycode, st.get_canonical_name(stycode))
                                                    for stycode in umls_ent.types]
                                        linked_entity["kb"] = data_source.name
                                        for stycode, styname in stycodes:
                                            linked_entity["id_code"] = data_source.entity_id
                                            linked_entity["type"] = styname
                                            linked_entity["type_id"] = stycode
                                    except Exception as e:
                                        exc_type, exc_obj, exc_tb = sys.exc_info()
                                        errorStr = re.sub(r"<class|>", "", str(exc_type)).strip()
                                        if errorStr not in errorsInAnnotation:
                                            errorsInAnnotation[errorStr] = [str(e)]
                                        else:
                                            errorsInAnnotation[errorStr].append(str(e))
                                        logging.warning(
                                            "There must be an entity whose code didn't get linked to the UMLS KB")
                                if data_source.name.lower() == 'snomedct_us':
                                    linked_entity["kb"] = data_source.name
                                    linked_entity["id_code"] = data_source.entity_id
                                if linked_entity:
                                    entity_annotation["linked_entities"].append(linked_entity)
                        sent_entities.append(entity_annotation)
                else:
                    errorSentences.append((i, doc))
                    logging.warning("Text analytics api didn't process this sentence")

                # for relation in doc.entity_relations:
                #   print(f"Relation of type: {relation.relation_type} has the following roles")
                #   for role in relation.roles:
                #       print(f"...Role '{role.name}' with entity '{role.entity.text}'")

                sent_id += 1

            if errorSentences:
                errorsInAnnotation[d] = errorSentences
            # print(sent_entities)
            document_batch_entities.extend(sent_entities)
            document_batch_sents.extend([i.split() for i in document])
        doc_ann["Entities"] = document_batch_entities
        doc_ann["Sents"] = document_batch_sents
        dataset_ann.append(doc_ann)

    return dataset_ann, errorsInAnnotation

def main(args):
    data_read = pd.read_csv(args.data, low_memory=False)
    data = data_read["CLEANED_TEXT"].tolist()
    documents = spacy_model.pipe(data)
    documents_segmented = []
    for doc in documents:
      docs = []
      for d in doc.sents:
        docs.append(" ".join([i.text for i in d]))
      documents_segmented.append(docs)

    if args.framework == 'microsoft':
        dataset, errors = ms_annotation(documents_segmented)
        dest = args.dest
        if not os.path.exists(dest):
            dest = os.makedirs(dest)

        with open(dest+'/ms_ann.json', 'w') as ms, open(dest+'/errors_in_annotation.json', 'w') as er:
            json.dump(dataset_ann, ms)
            json.dump(errorsInAnnotation, er)
            ms.close()
            er.close()

if __name__ == '__main__':
    par = argparse.ArgumentParser()
    par.add_argument('--data', default='../mimic-iii/cleaned/NOTESEVENTS_CLEANED.csv', help='location of the data')
    par.add_argument('--framework', default='spacy', help='spacy/cogstack/msanalytics')
    par.add_argument('--dest', default='../anns/spacy', help='location of the data')
    par.add_argument('--annotation_size', type=str, help='number of documents to annotate')
    par.add_argument('--model', default='en_core_sci_md', help='scispacy biomedical model')
    par.add_argument('--annotate_type', default='entities', help="what to annotate 'entities', or 'relations' or 'entities relations'")
    par.add_argument('--kg', type=str, default='umls', help='which knoiwledge graph is linked to for annotation')
    par.add_argument('--creds', default='text_analytics_credentials.yml', help='source of credentials to access azure language service')
    args = par.parse_args()

    #specifying microsoft text analytics language service parameters
    creds = yaml.safe_load(open(args.creds, 'r'))
    endpoint = creds['Subscription']['endpoint']
    key = creds['Subscription']['key1']
    text_analytics_client = TextAnalyticsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )

    #specifying spacy and scispacy model parameters
    disabled_modules = ["tagger", "parser"] if args.framework == "microsoft" else ["parser"]
    spacy_model = spacy.load(args.model)
    # spacy_model.add_pipe("scispacy_linker", config={"linker_name": "umls"})
    # spacy_model.add_pipe('senticizer')
    # kb = UmlsKnowledgeBase()
    # st = kb.semantic_type_tree

    #specifying
    main(args)