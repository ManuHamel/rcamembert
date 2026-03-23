#' Masked cross-entropy loss for token classification
#'
#' @param logits Logits tensor.
#' @param labels Labels tensor.
#' @param ignore_index Label value ignored in loss.
#' @return A torch tensor.
#' @export
masked_token_ce_loss <- function(logits, labels, ignore_index = -100L) {
  num_labels <- logits$size(3)
  logits2 <- logits$view(c(-1, num_labels))
  labels2 <- labels$view(c(-1))

  torch::nnf_cross_entropy(
    input = logits2,
    target = labels2,
    ignore_index = ignore_index
  )
}
