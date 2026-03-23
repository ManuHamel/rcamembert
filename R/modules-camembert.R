CamembertEmbeddings <- torch::nn_module(
  "CamembertEmbeddings",

  initialize = function(config) {
    self$word_embeddings <- torch::nn_embedding(
      num_embeddings = config$vocab_size,
      embedding_dim = config$hidden_size,
      padding_idx = config$pad_token_id
    )

    self$position_embeddings <- torch::nn_embedding(
      num_embeddings = config$max_position_embeddings,
      embedding_dim = config$hidden_size
    )

    self$layer_norm <- torch::nn_layer_norm(
      normalized_shape = config$hidden_size,
      eps = config$layer_norm_eps
    )

    self$dropout <- torch::nn_dropout(config$hidden_dropout_prob)
    self$pad_token_id <- config$pad_token_id
  },

  create_position_ids = function(input_ids) {
    mask <- input_ids$ne(self$pad_token_id)$to(dtype = torch::torch_long())
    incremental <- torch::torch_cumsum(mask, dim = 2)
    incremental * mask + self$pad_token_id
  },

  forward = function(input_ids) {
    position_ids <- self$create_position_ids(input_ids)
    x <- self$word_embeddings(input_ids) + self$position_embeddings(position_ids)
    x <- self$layer_norm(x)
    x <- self$dropout(x)
    x
  }
)

CamembertSelfAttention <- torch::nn_module(
  "CamembertSelfAttention",

  initialize = function(config) {
    stopifnot(config$hidden_size %% config$num_attention_heads == 0)

    self$num_attention_heads <- config$num_attention_heads
    self$attention_head_size <- as.integer(config$hidden_size / config$num_attention_heads)
    self$all_head_size <- config$hidden_size

    self$query <- torch::nn_linear(config$hidden_size, self$all_head_size)
    self$key   <- torch::nn_linear(config$hidden_size, self$all_head_size)
    self$value <- torch::nn_linear(config$hidden_size, self$all_head_size)

    self$dropout <- torch::nn_dropout(config$attention_probs_dropout_prob)
  },

  transpose_for_scores = function(x) {
    new_shape <- c(
      x$size(1),
      x$size(2),
      self$num_attention_heads,
      self$attention_head_size
    )
    x$view(new_shape)$permute(c(1, 3, 2, 4))
  },

  forward = function(hidden_states, attention_mask = NULL) {
    query_layer <- self$transpose_for_scores(self$query(hidden_states))
    key_layer   <- self$transpose_for_scores(self$key(hidden_states))
    value_layer <- self$transpose_for_scores(self$value(hidden_states))

    attention_scores <- torch::torch_matmul(
      query_layer,
      key_layer$transpose(3, 4)
    ) / sqrt(self$attention_head_size)

    if (!is.null(attention_mask)) {
      extended_mask <- attention_mask$
        to(dtype = torch::torch_float())$
        unsqueeze(2)$
        unsqueeze(2)

      additive_mask <- (1 - extended_mask) * -10000
      attention_scores <- attention_scores + additive_mask
    }

    attention_probs <- torch::nnf_softmax(attention_scores, dim = 4)
    attention_probs <- self$dropout(attention_probs)

    context_layer <- torch::torch_matmul(attention_probs, value_layer)
    context_layer <- context_layer$permute(c(1, 3, 2, 4))

    context_layer$contiguous()$view(c(
      hidden_states$size(1),
      hidden_states$size(2),
      self$all_head_size
    ))
  }
)

CamembertSelfOutput <- torch::nn_module(
  "CamembertSelfOutput",

  initialize = function(config) {
    self$dense <- torch::nn_linear(config$hidden_size, config$hidden_size)
    self$layer_norm <- torch::nn_layer_norm(config$hidden_size, eps = config$layer_norm_eps)
    self$dropout <- torch::nn_dropout(config$hidden_dropout_prob)
  },

  forward = function(hidden_states, input_tensor) {
    hidden_states <- self$dense(hidden_states)
    hidden_states <- self$dropout(hidden_states)
    self$layer_norm(hidden_states + input_tensor)
  }
)

CamembertAttention <- torch::nn_module(
  "CamembertAttention",

  initialize = function(config) {
    self$self_attn <- CamembertSelfAttention(config)
    self$output <- CamembertSelfOutput(config)
  },

  forward = function(hidden_states, attention_mask = NULL) {
    attn_out <- self$self_attn(hidden_states, attention_mask)
    self$output(attn_out, hidden_states)
  }
)

CamembertIntermediate <- torch::nn_module(
  "CamembertIntermediate",

  initialize = function(config) {
    self$dense <- torch::nn_linear(config$hidden_size, config$intermediate_size)
  },

  forward = function(hidden_states) {
    gelu(self$dense(hidden_states))
  }
)

CamembertOutput <- torch::nn_module(
  "CamembertOutput",

  initialize = function(config) {
    self$dense <- torch::nn_linear(config$intermediate_size, config$hidden_size)
    self$layer_norm <- torch::nn_layer_norm(config$hidden_size, eps = config$layer_norm_eps)
    self$dropout <- torch::nn_dropout(config$hidden_dropout_prob)
  },

  forward = function(hidden_states, input_tensor) {
    hidden_states <- self$dense(hidden_states)
    hidden_states <- self$dropout(hidden_states)
    self$layer_norm(hidden_states + input_tensor)
  }
)

CamembertLayer <- torch::nn_module(
  "CamembertLayer",

  initialize = function(config) {
    self$attention <- CamembertAttention(config)
    self$intermediate <- CamembertIntermediate(config)
    self$output <- CamembertOutput(config)
  },

  forward = function(hidden_states, attention_mask = NULL) {
    attention_output <- self$attention(hidden_states, attention_mask)
    intermediate_output <- self$intermediate(attention_output)
    self$output(intermediate_output, attention_output)
  }
)

CamembertEncoder <- torch::nn_module(
  "CamembertEncoder",

  initialize = function(config) {
    self$num_hidden_layers <- config$num_hidden_layers
    self$layers <- torch::nn_module_list(
      lapply(seq_len(config$num_hidden_layers), function(i) CamembertLayer(config))
    )
  },

  forward = function(hidden_states, attention_mask = NULL) {
    for (i in seq_len(self$num_hidden_layers)) {
      hidden_states <- self$layers[[i]](hidden_states, attention_mask)
    }
    hidden_states
  }
)

CamembertPooler <- torch::nn_module(
  "CamembertPooler",

  initialize = function(config) {
    self$dense <- torch::nn_linear(config$hidden_size, config$hidden_size)
  },

  forward = function(hidden_states) {
    first_token <- hidden_states[, 1, ]
    torch::torch_tanh(self$dense(first_token))
  }
)

CamembertModel <- torch::nn_module(
  "CamembertModel",

  initialize = function(config) {
    self$config <- config
    self$embeddings <- CamembertEmbeddings(config)
    self$encoder <- CamembertEncoder(config)
    self$pooler <- CamembertPooler(config)
  },

  forward = function(input_ids, attention_mask = NULL) {
    embedding_output <- self$embeddings(input_ids)
    sequence_output <- self$encoder(embedding_output, attention_mask)
    pooled_output <- self$pooler(sequence_output)

    list(
      last_hidden_state = sequence_output,
      pooler_output = pooled_output
    )
  }
)

CamembertForTokenClassification <- torch::nn_module(
  "CamembertForTokenClassification",

  initialize = function(config) {
    self$config <- config
    self$camembert <- CamembertModel(config)
    self$dropout <- torch::nn_dropout(config$hidden_dropout_prob)
    self$classifier <- torch::nn_linear(config$hidden_size, config$num_labels)
  },

  forward = function(input_ids, attention_mask = NULL, labels = NULL) {
    outputs <- self$camembert(input_ids, attention_mask)
    sequence_output <- self$dropout(outputs$last_hidden_state)
    logits <- self$classifier(sequence_output)

    loss <- NULL
    if (!is.null(labels)) {
      loss <- masked_token_ce_loss(logits, labels)
    }

    list(
      loss = loss,
      logits = logits,
      last_hidden_state = outputs$last_hidden_state
    )
  }
)
