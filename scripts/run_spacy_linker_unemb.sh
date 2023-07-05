# bash scripts to use microsoft language text analytics to annotate data
python ../src/scisp_ann.py --data ../../mimic-iii/cleaned/NOTESEVENTS_CLEANED_TEXT_ONLY.csv \
--annotation_size '0 5' \
--dest '../anns/scispacy/model_linker_unembedded' \
--model 'en_core_sci_sm'