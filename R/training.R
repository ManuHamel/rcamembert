#' Train a NER model
#'
#' @param model Model object.
#' @param train_dl Torch dataloader.
#' @param epochs Number of epochs.
#' @param lr Learning rate.
#' @param device Torch device.
#' @return Invisibly returns the model.
#' @export
train_ner_model <- function(model,
                            train_dl,
                            epochs = 5,
                            lr = 3e-5,
                            device = NULL) {
  if (is.null(device)) {
    device <- if (torch::cuda_is_available()) torch::torch_device("cuda") else torch::torch_device("cpu")
  }

  model$to(device = device)
  optimizer <- torch::optim_adamw(model$parameters, lr = lr)

  for (epoch in seq_len(epochs)) {
    model$train()
    total_loss <- 0

    coro::loop(for (batch in train_dl) {
      optimizer$zero_grad()

      input_ids <- batch$input_ids$to(device = device)
      attention_mask <- batch$attention_mask$to(device = device)
      labels <- batch$labels$to(device = device)

      out <- model(
        input_ids = input_ids,
        attention_mask = attention_mask,
        labels = labels
      )

      loss <- out$loss
      loss$backward()
      optimizer$step()

      total_loss <- total_loss + loss$item()
    })

    cat(sprintf("Epoch %d/%d - loss: %.4f\n", epoch, epochs, total_loss))
  }

  invisible(model)
}
