#' GELU activation
#'
#' @param x A torch tensor.
#' @return A torch tensor.
#' @export
gelu <- function(x) {
  0.5 * x * (1 + torch::torch_erf(x / sqrt(2)))
}

#' Create directory if needed
#'
#' @param path Directory path.
#' @return Invisibly returns the path.
#' @export
dir_create_if_needed <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
  }
  invisible(path)
}

#' Download a file if missing
#'
#' @param url Source URL.
#' @param dest Destination file path.
#' @param verbose Whether to print progress.
#' @return Invisibly returns destination path.
#' @export
download_file_if_missing <- function(url, dest, verbose = TRUE) {
  dir_create_if_needed(dirname(dest))
  if (!file.exists(dest)) {
    if (verbose) message("Downloading: ", basename(dest))
    httr2::request(url) |>
      httr2::req_perform(path = dest)
  }
  invisible(dest)
}

#' Download a file from Hugging Face repo
#'
#' @param repo_id Hugging Face repo id.
#' @param filename File to download.
#' @param cache_dir Cache directory.
#' @param verbose Whether to print progress.
#' @return Local file path.
#' @export
hf_download <- function(repo_id, filename, cache_dir = "hf-cache", verbose = TRUE) {
  repo_path <- gsub("/", "--", repo_id, fixed = TRUE)
  repo_cache <- file.path(cache_dir, repo_path)
  dir_create_if_needed(repo_cache)

  dest <- file.path(repo_cache, filename)
  url <- sprintf("https://huggingface.co/%s/resolve/main/%s", repo_id, filename)

  download_file_if_missing(url, dest, verbose = verbose)
  dest
}

#' Read JSON file
#'
#' @param path JSON file path.
#' @return Parsed R object.
#' @export
read_json_file <- function(path) {
  jsonlite::fromJSON(path, simplifyVector = TRUE)
}

#' Pretty-format extracted entities
#'
#' @param entities_df Data frame of entities.
#' @return Character string.
#' @export
format_entities_pretty <- function(entities_df) {
  if (nrow(entities_df) == 0) {
    return("Aucune entité détectée.")
  }

  lines <- apply(entities_df, 1, function(row) {
    sprintf("- [%s] %s", row[["type"]], row[["text"]])
  })

  paste(lines, collapse = "\n")
}
