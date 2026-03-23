#' Build default NER labels
#'
#' @return A list with label2id and id2label.
#' @export
build_ner_labels <- function() {
  label2id <- c(
    "O"      = 1L,
    "B-PER"  = 2L,
    "I-PER"  = 3L,
    "B-ORG"  = 4L,
    "I-ORG"  = 5L,
    "B-LOC"  = 6L,
    "I-LOC"  = 7L
  )

  list(
    label2id = label2id,
    id2label = names(label2id)
  )
}
