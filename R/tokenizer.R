#' Load CamemBERT tokenizer
#'
#' @param repo_id Hugging Face repo id.
#' @param cache_dir Cache directory.
#' @param verbose Whether to print progress.
#' @return A tokenizer object.
#' @export
camembert_tokenizer <- function(repo_id = "almanach/camembert-base",
                                cache_dir = "hf-cache",
                                verbose = TRUE) {
  spm_path <- hf_download(repo_id, "sentencepiece.bpe.model", cache_dir, verbose = verbose)
  model <- sentencepiece::sentencepiece_load_model(spm_path)

  list(
    model = model,
    special_ids = list(
      cls_token_id = 1L,
      pad_token_id = 2L,
      sep_token_id = 3L,
      unk_token_id = 4L,
      mask_token_id = 32005L
    )
  )
}
