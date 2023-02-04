# Phonetic Word Embeddings Suite (`PWESuite`)

Evaluation suite for phonetic (phonological) word embeddings and an additional model based on Panphone distance learning.

The suite contains the following tasks:
- Correlation with human sound similarity judgement
- Correlation with articulatory distance
- Nearest neighbour retrieval
- Rhyme detection
- Cognate detection
- Sound analogies

Run `pip3 install -e .` to install this repository and its dependencies.

## Embedding evaluation

In order to run all the evaluations, you first need to run the embedding on provided words.
These are found in `data/multi.tsv` (tab-separated values) in the format of:
```
word_ortho	word_ipa	language	word_purpose	pronunciation_information
```
Not all of the informations are provided for all languages, though `word_ortho` and `word_ipa` is guaranteed to be present.
Specifically, currently only English words contain all the aforementioned fields.
You can download the `multi.tsv` file from TODO (huggingface) or generate it locally by running `main/prepare_data.sh`.

After running the embedding on each word, save it as either a Pickle or NPZ. 
The data structure can be either (1) list of list or numpy arrays or (2) numpy array.
The loader will automatically parse the file and check that the dimensions are consistent.

After this, you are all set to run all the evaluations using `./suite_evaluation/eval_all.py --embd your_embd.pkl`.
Alternatively, you can invoke individual tasks: `./suite_evaluation/eval_{correlations,human_similarity,retrieval,analogy,rhyme,cognate}.py`.