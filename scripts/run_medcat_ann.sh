# bash scripts to use microsoft language text analytics to annotate data
python annotate.py --data_dir ../NOTESEVENTS_CLEANED_5.csv \
--annotation_size '0 20' \
--dest '../anns' \
--model '../models/medmen_wstatus_2021_oct.zip' 