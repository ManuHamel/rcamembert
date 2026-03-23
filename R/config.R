#' Build CamemBERT configuration
#'
#' @param vocab_size Vocabulary size.
#' @param max_position_embeddings Maximum number of position embeddings.
#' @param type_vocab_size Type vocabulary size.
#' @param hidden_size Hidden size.
#' @param intermediate_size Intermediate size.
#' @param num_attention_heads Number of attention heads.
#' @param num_hidden_layers Number of transformer layers.
#' @param hidden_dropout_prob Hidden dropout.
#' @param attention_probs_dropout_prob Attention dropout.
#' @param layer_norm_eps Layer norm epsilon.
#' @param pad_token_id Padding token id.
#' @param bos_token_id BOS token id.
#' @param eos_token_id EOS token id.
#' @param num_labels Number of output labels.
#' @return A configuration list.
#' @export
camembert_config <- function(
    vocab_size = 32005L,
    max_position_embeddings = 514L,
    type_vocab_size = 1L,
    hidden_size = 768L,
    intermediate_size = 3072L,
    num_attention_heads = 12L,
    num_hidden_layers = 12L,
    hidden_dropout_prob = 0.1,
    attention_probs_dropout_prob = 0.1,
    layer_norm_eps = 1e-5,
    pad_token_id = 2L,
    bos_token_id = 1L,
    eos_token_id = 3L,
    num_labels = 7L) {

  list(
    vocab_size = as.integer(vocab_size),
    max_position_embeddings = as.integer(max_position_embeddings),
    type_vocab_size = as.integer(type_vocab_size),
    hidden_size = as.integer(hidden_size),
    intermediate_size = as.integer(intermediate_size),
    num_attention_heads = as.integer(num_attention_heads),
    num_hidden_layers = as.integer(num_hidden_layers),
    hidden_dropout_prob = hidden_dropout_prob,
    attention_probs_dropout_prob = attention_probs_dropout_prob,
    layer_norm_eps = layer_norm_eps,
    pad_token_id = as.integer(pad_token_id),
    bos_token_id = as.integer(bos_token_id),
    eos_token_id = as.integer(eos_token_id),
    num_labels = as.integer(num_labels)
  )
}

#' Load CamemBERT config from Hugging Face
#'
#' @param repo_id Hugging Face repo id.
#' @param cache_dir Cache directory.
#' @param num_labels Number of labels.
#' @param verbose Whether to print progress.
#' @return A configuration list.
#' @export
load_camembert_config_from_hf <- function(repo_id = "almanach/camembert-base",
                                          cache_dir = "hf-cache",
                                          num_labels = 7L,
                                          verbose = TRUE) {
  cfg_path <- hf_download(repo_id, "config.json", cache_dir, verbose = verbose)
  cfg_json <- read_json_file(cfg_path)

  camembert_config(
    vocab_size = cfg_json$vocab_size + 1L,
    max_position_embeddings = cfg_json$max_position_embeddings + 1L,
    type_vocab_size = if (!is.null(cfg_json$type_vocab_size)) cfg_json$type_vocab_size else 1L,
    hidden_size = cfg_json$hidden_size,
    intermediate_size = cfg_json$intermediate_size,
    num_attention_heads = cfg_json$num_attention_heads,
    num_hidden_layers = cfg_json$num_hidden_layers,
    hidden_dropout_prob = cfg_json$hidden_dropout_prob,
    attention_probs_dropout_prob = cfg_json$attention_probs_dropout_prob,
    layer_norm_eps = cfg_json$layer_norm_eps,
    pad_token_id = cfg_json$pad_token_id + 1L,
    bos_token_id = cfg_json$bos_token_id + 1L,
    eos_token_id = cfg_json$eos_token_id + 1L,
    num_labels = num_labels
  )
}
