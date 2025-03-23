#!/bin/bash

for page in {1..1375}; do
    template_index=$(( (page - 1) % 5 + 1 ))
    template="CUSTOM_TEMPLATE0$template_index"
	echo python 06-distilabel_medical_with_qa.py \
	--url http://127.0.0.1:8000/v1 \
	--dataset c00cjz00/Medical-R1-Distill-Data \
	--datasetconfig default --datasetsplit train \
	--questioncolumn question \
	--answercolumn response \
	--model c00cjz00/phi-4-14b-it-offon-R1-m22k \
	--page $page \
	--pagesize 1024 \
	--batchsize 64 \
	--temperature 0.6 \
	--maxnewtokens 4096 \
	--template "$template"
	
done
