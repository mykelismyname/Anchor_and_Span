import numpy as np

#create spans of batches that would be used toselect chunks of documents that can be processed by Text analytics without which ratelimitting errors
def batching_for_textanalyticsclinet(batch, size):
    x, y = 0, batch
    batch_list = []
    for i in range(0, size, batch):
        if i + batch >= size:
            batch_list.append((x, x + (size - i)))
            break
        batch_list.append((x, y))
        x = y
        y = y + batch
    return batch_list

#find the token id/position of an entity given its char offset from start of a sentence and the sentence
def fecth_entitis_span_pos(entity, sentence):
    sentence_tokenized = dict([(i, j) for i, j in enumerate(sentence.split())])
    token_ids_cumulative_length = {}
    char_offset, entity_len = entity.offset, entity.length
    curr_offset = 0
    span, end_span_found = [], False
    if char_offset == 0:
        span.append(0)

    for i, j in sentence_tokenized.items():
        if curr_offset >= char_offset + entity_len:
            span.append(i)
            end_span_found = True
        else:
            curr_offset = curr_offset + len(j) + 1
            if curr_offset == char_offset:
                span.append(i + 1)
        if end_span_found == True:
            if len(span) != 2:
                print("Failed span", entity.text, span)
            try:
                assert len(span) == 2
                break
            except Exception as e:
                return span
    return span