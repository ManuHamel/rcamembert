#' Load Hugging Face weights into model
#'
#' Placeholder extension point for mapping Hugging Face tensors
#' to R torch modules.
#'
#' @param model A token classification model.
#' @param repo_id Hugging Face repo id.
#' @param cache_dir Cache directory.
#' @param verbose Whether to print progress.
#' @return Model.
#' @export
load_hf_weights_into_model <- function(model,
                                       repo_id = "almanach/camembert-base",
                                       cache_dir = "hf-cache",
                                       verbose = TRUE) {
  if (verbose) {
    message("No Hugging Face weight loader is implemented in this version.")
    message("Extension point available in load_hf_weights_into_model().")
  }

  model
}
