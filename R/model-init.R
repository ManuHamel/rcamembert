#' Initialize CamemBERT NER model
#'
#' @param num_labels Number of output labels.
#' @param repo_id Hugging Face repo id.
#' @param cache_dir Cache directory.
#' @param device Torch device.
#' @param load_hf_weights Whether to load HF weights.
#' @param verbose Whether to print progress.
#' @return A list with model, config, and device.
#' @export
initialize_camembert_ner_model <- function(num_labels,
                                           repo_id = "almanach/camembert-base",
                                           cache_dir = "hf-cache",
                                           device = NULL,
                                           load_hf_weights = FALSE,
                                           verbose = TRUE) {
  if (is.null(device)) {
    device <- if (torch::cuda_is_available()) torch::torch_device("cuda") else torch::torch_device("cpu")
  }

  config <- load_camembert_config_from_hf(
    repo_id = repo_id,
    cache_dir = cache_dir,
    num_labels = num_labels,
    verbose = verbose
  )

  model <- CamembertForTokenClassification(config)

  if (load_hf_weights) {
    model <- load_hf_weights_into_model(
      model = model,
      repo_id = repo_id,
      cache_dir = cache_dir,
      verbose = verbose
    )
  } else if (verbose) {
    message("Model initialized without explicit Hugging Face weights.")
  }

  model$to(device = device)

  list(
    model = model,
    config = config,
    device = device
  )
}
