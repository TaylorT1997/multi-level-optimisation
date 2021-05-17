# TODO

## High Priority

- [X] Sentence classification on datasets (baseline model)
- [X] Develop torch dataset for sentence classification
- [X] Develop torch dataset for token classification
- [X] Create model with token and sentence supervision
- [ ] Full run of token model

## Medium Priority

- [X] Develop scripts that automate the entire data processing pipeline
- [X] Setup wandb for experiment tracking
- [X] Setup remote GPU for training
- [X] Train baseline sentence level models
- [X] Get access to eval file from CoNLL-10 authors
- [X] Implement early stopping
- [X] Add regularisation losses
- [X] Add BERT losses
- [X] Run sentence classification on DeBERTa
- [X] Add LR scheduler
- [X] State of the art comparison
- [ ] Start on background report
- [X] Add quadratic soft attention
- [ ] Add option for CRT
- [X] Add config file
- [ ] Evaluate models
- [ ] Split wi locness to 80 20

## Low Priority

- [ ] Overleaf for papers
- [X] Write up data processing progress
- [ ] Write up sentence dataset progress
- [X] Share repo
- [X] Add sigmoid to make prediction
- [X] Add validation set
- [X] Add precision recall f1
- [X] Add more configs
- [X] Add console prints
- [ ] Functionise logging metrics
- [X] Fix CoNLL-10 train download and processing
- [ ] Read and summarise 'core' papers
- [X] Remove CLS and SEP tokens
- [ ] Add more schedulers and optimizers
- [X] Apply mask to normalisation of attention
- [X] Add option to use CLS token
- [ ] Add F0.5 measure
- [ ] Add token logging

# Blockers

## High Priority


## Medium Priority

- [X] Find task2_eval xml file for CONLL-10 

## Low Priority


# Questions

## High Priority

- [ ] Is SUM SUM too harsh a loss for the tokens?
- [X] How important is training the pretrained encoder?

## Medium Priority

- [X] Is it better to add unknown words to vocab or split into tokens?
- [X] How are split tokens dealt with during testing?
- [X] How exactly should dev splits be used? What is the benefit? Do we train on train+dev?
- [X] Which token labelling approach is preferrable? All or First?
- [X] Can we completely ignore CLS and SEP tokens in sentence classification?
- [X] Could we do everything directly from BERT?
- [X] Should we normalise the losses by batch size?
- [X] No toxic results?
- [X] What is F0.5?
- [ ] Can I know the max seq len of the validation set?
- [ ] Can I directly optimise for f1?

## Low Priority

- [X] What should I be reading for lit review? BERT? DeBERTA? Domain specific papers?
- [X] Can the model use the fact that a word has been split into multiple word parts to predict that it is a mistake?

# Notes

- Remember to change the size of the embedding matrix after increasing the number of tokens! (https://github.com/huggingface/transformers/issues/1413#issuecomment-538083512, https://github.com/huggingface/tokenizers/issues/247#issuecomment-675458087)
