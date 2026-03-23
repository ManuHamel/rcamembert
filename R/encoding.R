#' Encode a NER example
#'
#' @param words Character vector of words.
#' @param labels Character vector of labels.
#' @param tokenizer Tokenizer object.
#' @param label2id Named integer vector.
#' @param max_length Maximum sequence length.
#' @param ignore_index Ignore index for non-first subword labels.
#' @return A list with encoded tensors/vectors.
#' @export
encode_ner_example <- function(words,
                               labels,
                               tokenizer,
                               label2id,
                               max_length = 64L,
                               ignore_index = -100L) {
  stopifnot(length(words) == length(labels))

  cls_id <- tokenizer$special_ids$cls_token_id
  sep_id <- tokenizer$special_ids$sep_token_id
  pad_id <- tokenizer$special_ids$pad_token_id
  unk_id <- tokenizer$special_ids$unk_token_id

  input_ids <- c(cls_id)
  out_labels <- c(ignore_index)
  token_strings <- c("<s>")
  word_ids <- c(NA_integer_)

  for (i in seq_along(words)) {
    word <- words[[i]]
    lab  <- labels[[i]]

    sub_ids <- sentencepiece::sentencepiece_encode(
      model = tokenizer$model,
      x = word,
      type = "ids",
      nbest = 1L,
      alpha = 0
    )[[1]]

    sub_toks <- sentencepiece::sentencepiece_encode(
      model = tokenizer$model,
      x = word,
      type = "subwords",
      nbest = 1L,
      alpha = 0
    )[[1]]

    if (length(sub_ids) == 0) {
      sub_ids <- unk_id
      sub_toks <- "<unk>"
    } else {
      sub_ids <- as.integer(sub_ids) + 1L
    }

    sub_labels <- c(label2id[[lab]], rep(ignore_index, max(0L, length(sub_ids) - 1L)))

    input_ids <- c(input_ids, sub_ids)
    out_labels <- c(out_labels, sub_labels)
    token_strings <- c(token_strings, sub_toks)
    word_ids <- c(word_ids, rep(i, length(sub_ids)))
  }

  input_ids <- c(input_ids, sep_id)
  out_labels <- c(out_labels, ignore_index)
  token_strings <- c(token_strings, "</s>")
  word_ids <- c(word_ids, NA_integer_)

  if (length(input_ids) > max_length) {
    input_ids <- input_ids[seq_len(max_length)]
    out_labels <- out_labels[seq_len(max_length)]
    token_strings <- token_strings[seq_len(max_length)]
    word_ids <- word_ids[seq_len(max_length)]
  }

  attention_mask <- rep(1L, length(input_ids))

  if (length(input_ids) < max_length) {
    n_pad <- max_length - length(input_ids)
    input_ids <- c(input_ids, rep(pad_id, n_pad))
    attention_mask <- c(attention_mask, rep(0L, n_pad))
    out_labels <- c(out_labels, rep(ignore_index, n_pad))
    token_strings <- c(token_strings, rep("<pad>", n_pad))
    word_ids <- c(word_ids, rep(NA_integer_, n_pad))
  }

  list(
    input_ids = as.integer(input_ids),
    attention_mask = as.integer(attention_mask),
    labels = as.integer(out_labels),
    tokens = token_strings,
    word_ids = as.integer(word_ids)
  )
}

#' Prepare inference example
#'
#' @param text Input text.
#' @param tokenizer Tokenizer object.
#' @param label2id Named vector label-to-id.
#' @param max_length Maximum sequence length.
#' @return Encoded example ready for prediction.
#' @export
prepare_inference_example <- function(text,
                                      tokenizer,
                                      label2id,
                                      max_length = 64L) {
  words <- unlist(strsplit(text, "\\s+"))
  words <- words[nzchar(words)]

  enc <- encode_ner_example(
    words = words,
    labels = rep("O", length(words)),
    tokenizer = tokenizer,
    label2id = label2id,
    max_length = max_length
  )

  list(
    words = words,
    tokens = enc$tokens,
    word_ids = enc$word_ids,
    input_ids = torch::torch_tensor(matrix(enc$input_ids, nrow = 1), dtype = torch::torch_long()),
    attention_mask = torch::torch_tensor(matrix(enc$attention_mask, nrow = 1), dtype = torch::torch_long())
  )
}
