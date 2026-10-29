#!/usr/bin/env Rscript
# Chord diagram of the mean number of transitions between states.
#
# Reads the ordered pair jump registers (c_<pair-prefix>.<from>.<to>[1]) from a
# BEAST log.  The legacy geographic analysis uses pair prefix "Location".
#
# Usage:
#   Rscript circlize.R --log results/dta.log --output results/dta.circlize.svg \
#       --pair-prefix Location

suppressPackageStartupMessages({
  library(circlize)
  library(RColorBrewer)
})

parse_args <- function(args) {
  parsed <- list()
  i <- 1
  while (i <= length(args)) {
    token <- args[[i]]
    if (grepl("^--[A-Za-z0-9-]+=", token)) {
      parsed[[sub("^--", "", sub("=.*$", "", token))]] <- sub("^[^=]*=", "", token)
      i <- i + 1
    } else if (grepl("^--", token)) {
      if (i == length(args)) stop(paste("missing value for", token), call. = FALSE)
      parsed[[sub("^--", "", token)]] <- args[[i + 1]]
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  parsed
}

opt <- parse_args(commandArgs(trailingOnly = TRUE))
if (is.null(opt$log)) stop("--log is required", call. = FALSE)
if (is.null(opt$output)) stop("--output is required", call. = FALSE)
pair_prefix <- if (is.null(opt[["pair-prefix"]])) "Location" else opt[["pair-prefix"]]
ext <- if (is.null(opt$ext)) ".svg" else opt$ext

data <- read.csv(opt$log, sep = "\t", comment.char = "#", header = TRUE, check.names = FALSE)

create_migration_matrix <- function(df, pair_prefix) {
  prefix <- paste0("c_", pair_prefix, ".")
  columns <- colnames(df)[startsWith(colnames(df), prefix)]
  columns <- columns[!startsWith(columns, paste0("c_", pair_prefix, ".total"))]
  if (length(columns) == 0) {
    stop(sprintf("no %s* columns found in the log", prefix), call. = FALSE)
  }

  # "c_Location.India.Koshi[1]" -> c("India", "Koshi")
  strip <- function(column) {
    cleaned <- sub("\\[[0-9]+\\]$", "", sub(prefix, "", column, fixed = TRUE))
    parts <- strsplit(cleaned, ".", fixed = TRUE)[[1]]
    if (length(parts) != 2) {
      stop(sprintf("cannot split '%s' into an origin and a destination", column), call. = FALSE)
    }
    parts
  }

  pairs <- lapply(columns, strip)
  states <- unique(unlist(pairs))
  migration_matrix <- matrix(
    0,
    nrow = length(states), ncol = length(states),
    dimnames = list(states, states)
  )
  for (i in seq_along(columns)) {
    origin <- pairs[[i]][1]
    destination <- pairs[[i]][2]
    migration_matrix[origin, destination] <- mean(df[[columns[i]]])
  }
  migration_matrix
}

mean_migs <- create_migration_matrix(data, pair_prefix)
print(mean_migs)

states <- sort(rownames(mean_migs))
n_states <- length(states)
if (n_states > 9) {
  warning("More than 9 states. Switching to a larger palette.")
  grid.col <- colorRampPalette(brewer.pal(12, "Set3"))(n_states)
} else {
  grid.col <- colorRampPalette(brewer.pal(9, "Set1"))(n_states)
}
names(grid.col) <- states
print(grid.col)

if (ext == ".svg" && requireNamespace("svglite", quietly = TRUE)) {
  svglite::svglite(opt$output, width = 6, height = 6)
} else if (ext == ".svg") {
  grDevices::svg(opt$output, width = 6, height = 6)
} else if (ext == ".pdf") {
  grDevices::pdf(opt$output, width = 6, height = 6)
} else {
  grDevices::png(opt$output, width = 6, height = 6, units = "in", res = 150)
}

arr.col <- data.frame(
  expand.grid(states, states),
  rep("black", times = n_states * n_states)
)

chordDiagram(
  as.matrix(mean_migs),
  transparency = 0.1,
  directional = 1,
  direction.type = c("arrows", "diffHeight"),
  link.arr.col = arr.col,
  link.arr.length = 0.1,
  link.arr.type = "big.arrow",
  annotationTrack = "grid",
  grid.col = grid.col
)

circos.trackPlotRegion(
  track.index = 1,
  bg.border = NA,
  panel.fun = function(x, y) {
    xlim <- get.cell.meta.data("xlim")
    sector.index <- get.cell.meta.data("sector.index")
    circos.text(x = mean(xlim), y = 2, labels = sector.index, facing = "bending", cex = 1)
  }
)

grDevices::dev.off()

if (!file.exists(opt$output)) {
  stop(
    sprintf(
      "the graphics device could not write %s; an SVG device needs a cairo-capable R",
      opt$output
    ),
    call. = FALSE
  )
}
