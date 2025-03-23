grep -rl '"raw_output_text_generation_0":null'  ~/.cache/distilabel/pipelines/tw-instruct-500k-demo_qa_*/steps_data | xargs rm -f
