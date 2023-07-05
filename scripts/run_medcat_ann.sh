# bash scripts to use microsoft language text analytics to annotate data
python ../src/medcat_ann.py --data_dir ../../mimic-iii/cleaned/NOTESEVENTS_CLEANED_TEXT_ONLY.csv \
--annotation_size '0 20' \
--dest '../anns' \
--medcat_model '../medcat_models/medmen_wstatus_2021_oct.zip' \
--multiprocessing