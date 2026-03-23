#' Predict named entities from text
#'
#' @param model A trained model.
#' @param tokenizer Tokenizer object.
#' @param text Input text.
#' @param label2id Label-to-id mapping.
#' @param id2label Id-to-label mapping.
#' @param max_length Maximum sequence length.
#' @param device Torch device.
#' @return A list with token, word, and entity predictions.
#' @export
predict_ner_text <- function(model,
                             tokenizer,
                             text,
                             label2id,
                             id2label,
                             max_length = 64L,
                             device = NULL) {
  if (is.null(device)) {
    device <- if (torch::cuda_is_available()) torch::torch_device("cuda") else torch::torch_device("cpu")
  }

  model$eval()
  model$to(device = device)

  ex <- prepare_inference_example(
    text = text,
    tokenizer = tokenizer,
    label2id = label2id,
    max_length = max_length
  )

  torch::with_no_grad({
    out <- model(
      input_ids = ex$input_ids$to(device = device),
      attention_mask = ex$attention_mask$to(device = device)
    )

    pred_ids <- as.integer(out$logits$argmax(dim = 3)$cpu()[1, ])
    pred_labels <- id2label[pred_ids]

    token_df <- data.frame(
      token = ex$tokens,
      word_id = ex$word_ids,
      pred_label = pred_labels,
      stringsAsFactors = FALSE
    )

    token_df <- token_df[token_df$token != "<pad>", , drop = FALSE]

    keep <- !is.na(token_df$word_id) & !duplicated(token_df$word_id)
    word_labels <- token_df$pred_label[keep]

    word_df <- data.frame(
      word = ex$words[seq_along(word_labels)],
      pred_label = word_labels,
      stringsAsFactors = FALSE
    )

    entities_df <- extract_entities_from_predictions(
      words = word_df$word,
      word_level_labels = word_df$pred_label
    )

    list(
      token_predictions = token_df,
      word_predictions = word_df,
      entities = entities_df,
      pretty_entities = format_entities_pretty(entities_df)
    )
  })
}
