rcamembert

rcamembertR is an R package that provides a CamemBERT-like implementation in R using torch for Named Entity Recognition (NER), including:

a SentencePiece-based tokenizer
a Transformer / CamemBERT-style encoder
a token classification head
helper functions to:
load configuration from Hugging Face
prepare datasets
train a model
run inference on French text
extract entities in a readable format

The goal of this package is to provide a 100% R-based foundation for experimenting with CamemBERT, without relying on a Python/reticulate pipeline for the core model logic.

Features
R implementation of the main components of a CamemBERT-like model:
embeddings
multi-head self-attention
Transformer blocks
encoder
token-level classification layer
Tokenization via sentencepiece
Preparation of NER datasets with word/subword alignment
Training of an NER model using torch
Prediction on raw text
Reconstruction of entities from BIO tags
Download of configuration and tokenizer files from Hugging Face
Full demo included

# Installation

```r
install.packages("remotes")
remotes::install_github("ManuHamel/rcamembert")
```

# Example 

```r
library(rcamembert)

repo_id <- "almanach/camembert-base"
cache_dir <- "hf-cache"
text_to_predict <- "Emmanuel Macron rencontre des chercheurs de Mila à Montréal."
epochs <- 5
lr <- 3e-5
max_train_length <- 32L
max_predict_length <- 64L
batch_size <- 2L
load_hf_weights <- FALSE
verbose <- TRUE

set.seed(42)
torch::torch_manual_seed(42)

device <- if (torch::cuda_is_available()) torch::torch_device("cuda") else torch::torch_device("cpu")
cat("Device:", device$type, "\n")

ner_labels <- build_ner_labels()
label2id <- ner_labels$label2id
id2label <- ner_labels$id2label

cat("Loading tokenizer...\n")
tokenizer <- camembert_tokenizer(
  repo_id = repo_id,
  cache_dir = cache_dir,
  verbose = verbose
)

cat("Initializing model...\n")
model_obj <- initialize_camembert_ner_model(
  num_labels = length(label2id),
  repo_id = repo_id,
  cache_dir = cache_dir,
  device = device,
  load_hf_weights = load_hf_weights,
  verbose = verbose
)

model <- model_obj$model

cat("Preparing training data...\n")
train_examples <- list(
  list(
    words  = c("Emmanuel", "Macron", "visite", "Montréal", "avec", "des", "chercheurs", "de", "Mila"),
    labels = c("B-PER", "I-PER", "O", "B-LOC", "O", "O", "O", "O", "B-ORG")
  ),
  list(
    words  = c("Jean", "Dupont", "travaille", "chez", "Airbus", "à", "Toulouse"),
    labels = c("B-PER", "I-PER", "O", "O", "B-ORG", "O", "B-LOC")
  ),
  list(
    words  = c("Sophie", "habite", "à", "Paris"),
    labels = c("B-PER", "O", "O", "B-LOC")
  ),
  list(
    words  = c("OpenAI", "ouvre", "un", "bureau", "à", "Montréal"),
    labels = c("B-ORG", "O", "O", "O", "O", "B-LOC")
  ),
  list(
    words  = c("Paul", "Martin", "rencontre", "Marie", "Curie", "à", "Lyon"),
    labels = c("B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-LOC")
  )
)

max_length <- 32L
encoded_train <- lapply(train_examples, function(ex) {
  encode_ner_example(
    words = ex$words,
    labels = ex$labels,
    tokenizer = tokenizer,
    label2id = label2id,
    max_length = max_length
  )
})

input_mat <- do.call(rbind, lapply(encoded_train, `[[`, "input_ids"))
attn_mat  <- do.call(rbind, lapply(encoded_train, `[[`, "attention_mask"))
label_mat <- do.call(rbind, lapply(encoded_train, `[[`, "labels"))

train_ds <- ner_dataset(
  input_ids = torch::torch_tensor(input_mat, dtype = torch::torch_long()),
  attention_mask = torch::torch_tensor(attn_mat, dtype = torch::torch_long()),
  labels = torch::torch_tensor(label_mat, dtype = torch::torch_long())
)

train_dl <- torch::dataloader(train_ds, batch_size = batch_size, shuffle = TRUE)

cat("Training...\n")
train_ner_model(
  model = model,
  train_dl = train_dl,
  epochs = epochs,
  lr = lr,
  device = device
)

cat("\nInput text:\n")
cat(text_to_predict, "\n")

pred <- predict_ner_text(
  model = model,
  tokenizer = tokenizer,
  text = text_to_predict,
  label2id = label2id,
  id2label = id2label,
  max_length = max_predict_length,
  device = device
)

cat("\nPredictions by word:\n")
print(pred$word_predictions)

cat("\nExtracted entities:\n")
cat(pred$pretty_entities, "\n")
```
