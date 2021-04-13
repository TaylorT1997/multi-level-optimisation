# TODO

## High Priority

- [X] Sentence classification on datasets (baseline model)
- [ ] Develop torch datasets

## Medium Priority

- [X] Develop scripts that automate the entire data processing pipeline
- [X] Setup wandb for experiment tracking
- [ ] Setup remote GPU for training

## Low Priority

- [ ] Google doc for papers
- [ ] Write up data processing progress
- [ ] Write up sentence dataset progress
- [X] Share repo
- [X] Add sigmoid to make prediction
- [X] Add validation set
- [ ] Add precision recall f1

# Blockers

## High Priority


## Medium Priority

- Find task2_eval xml file for CONLL-10 

## Low Priority


# Questions

## High Priority


## Medium Priority

- Is it better to add unknown words to vocab or split into tokens?
- How are split tokens dealt with during testing?
- How exactly should dev splits be used? What is the benefit? Do we train on train+dev?

## Low Priority

- What should I be reading for lit review? BERT? DeBERTA? Domain specific papers?


# Notes

- Remember to change the size of the embedding matrix after increasing the number of tokens! (https://github.com/huggingface/transformers/issues/1413#issuecomment-538083512, https://github.com/huggingface/tokenizers/issues/247#issuecomment-675458087)
