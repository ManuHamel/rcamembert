#' Extract entities from predicted BIO tags
#'
#' @param words Character vector of words.
#' @param word_level_labels Predicted BIO labels.
#' @return A data frame.
#' @export
extract_entities_from_predictions <- function(words, word_level_labels) {
  entities <- list()
  current_type <- NULL
  current_tokens <- character()

  flush_entity <- function() {
    if (!is.null(current_type) && length(current_tokens) > 0) {
      entities[[length(entities) + 1]] <<- list(
        type = current_type,
        text = paste(current_tokens, collapse = " ")
      )
    }
    current_type <<- NULL
    current_tokens <<- character()
  }

  for (i in seq_along(words)) {
    lab <- word_level_labels[i]
    tok <- words[i]

    if (lab == "O") {
      flush_entity()
      next
    }

    if (startsWith(lab, "B-")) {
      flush_entity()
      current_type <- sub("^B-", "", lab)
      current_tokens <- tok
      next
    }

    if (startsWith(lab, "I-")) {
      entity_type <- sub("^I-", "", lab)
      if (!is.null(current_type) && identical(current_type, entity_type)) {
        current_tokens <- c(current_tokens, tok)
      } else {
        flush_entity()
        current_type <- entity_type
        current_tokens <- tok
      }
      next
    }

    flush_entity()
  }

  flush_entity()

  if (length(entities) == 0) {
    return(data.frame(type = character(), text = character(), stringsAsFactors = FALSE))
  }

  do.call(
    rbind,
    lapply(entities, function(x) {
      data.frame(type = x$type, text = x$text, stringsAsFactors = FALSE)
    })
  )
}
