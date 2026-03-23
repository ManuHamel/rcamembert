#' Torch dataset for NER
#'
#' @export
ner_dataset <- torch::dataset(
  name = "ner_dataset",

  initialize = function(input_ids, attention_mask, labels) {
    self$input_ids <- input_ids
    self$attention_mask <- attention_mask
    self$labels <- labels
  },

  .getitem = function(i) {
    list(
      input_ids = self$input_ids[i, ..],
      attention_mask = self$attention_mask[i, ..],
      labels = self$labels[i, ..]
    )
  },

  .length = function() {
    self$input_ids$size(1)
  }
)
