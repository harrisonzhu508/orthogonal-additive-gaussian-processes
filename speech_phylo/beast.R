library(doParallel)
library(dplyr)
library(tidyr)
library(TreeDist)

source("speech_phylo/read.annot.beast.R")


extract_beast_metadata <- function(file = "", tree = NULL) {
  if (!is.null(tree)) {
    if (!inherits(tree, "phylo")) {
      stop("argument 'tree' must be of mode 'phylo'")
    }
  } else if (file != "") {
    tree <- read.annot.beast(file)
  } else {
    stop("argument 'file' or 'tree' must be provided")
  }

  is_tip <- tree$edge[, 2] <= length(tree$tip.label)
  ordered_tips <- tree$edge[is_tip, 2]
  df <- tree$metadata |>
    mutate(language = tree$tip.label[ordered_tips[node]]) |>
    filter(!is.na(language)) |>
    mutate(across(everything(), ~ type.convert(., as.is = TRUE)))

  df
}

#' @examples
#' input_dir <- "data/beast/eab44e7f-54cc-4469-87d1-282cc81e02c2/sentence"
#' pattern <- "*\\_treeannotator_CCD.nex"
#' cores <- 16
#' extract_beast_metadata(input_dir, pattern, cores)
extract_beast_metadata2 <- function(files = "", trees = "", output_file = "", cores = 1) {
  if (!is.null(trees)) {
    if (!inherits(trees, "multiPhylo")) {
      stop("argument 'tree' must be of mode 'phylo'")
    }
    obj <- trees
  } else if (file != "") {
    obj <- files
  } else {
    stop("argument 'file' or 'tree' must be provided")
  }

  stopifnot(length(obj) > 0)

  registerDoParallel(cores = cores)
  print(paste("Using", cores, "core(s)", sep = " "))

  result <- foreach(i = seq_along(obj), .combine = rbind) %dopar% {
    if (inherits(obj[[i]], "phylo")) {
      extract_beast_metadata(tree = obj[[i]])
    } else {
      extract_beast_metadata(file = obj[[i]])
    }
  }

  if (output_file != "") {
    write.csv(result, output_file)
  } else {
    result
  }
}

extract_beast_dists <- function(input_dir, pattern, cores = 1) {
  files <- list.files(
    input_dir,
    pattern = pattern,
    full.names = TRUE
  )

  stopifnot(length(files) > 0)
  print(paste("Found", length(files), "files", sep = " "))

  registerDoParallel(cores = cores)
  print(paste("Using", cores, "core(s)", sep = " "))

  print("Reading trees...")
  trees <- foreach(i = seq_along(files)) %dopar% {
    read.annot.beast(files[i])
  }

  dists <- TreeDistance(trees)

  output_file <- file.path(input_dir, "_dists.Rdata")

  save(dists, file = output_file)

  print(paste("Done. View results at", output_file, sep = " "))
}
