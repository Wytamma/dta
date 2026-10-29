#!/usr/bin/env Rscript
# Renders a BEAST maximum clade credibility tree with the reconstructed states.
#
# The trait annotation tag is configurable so the same script serves the legacy
# geographic analysis (tag "location") and arbitrary genomic traits.
#
# Usage:
#   Rscript plotMCC.R --input mcc.nex --output-prefix results/dta.mcc.mean \
#       --trait-tag location --mrsd 2021-06-28 --ext .svg
#
# Only ggtree/treeio, ggplot2 and RColorBrewer are required (svglite is used
# for SVG output when it is installed, otherwise base R's cairo device is).

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggtree)
  library(treeio)
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
if (is.null(opt$input)) stop("--input is required", call. = FALSE)

output_prefix <- if (is.null(opt[["output-prefix"]])) "mcc.tree" else opt[["output-prefix"]]
trait_tag <- if (is.null(opt[["trait-tag"]])) "location" else opt[["trait-tag"]]
ext <- if (is.null(opt$ext)) ".png" else opt$ext
mrsd <- opt$mrsd
width <- if (is.null(opt$width)) 7 else as.numeric(opt$width)
height <- if (is.null(opt$height)) 7 else as.numeric(opt$height)

beast <- treeio::read.beast(opt$input)
annotations <- colnames(beast@data)

if (!trait_tag %in% annotations) {
  stop(
    sprintf(
      "annotation '%s' not found in %s; available: %s",
      trait_tag, opt$input, paste(annotations, collapse = ", ")
    ),
    call. = FALSE
  )
}

# BEAST-safe state codes (s0, s1, ...) are mapped back to the user's labels
if (!is.null(opt$labels) && file.exists(opt$labels)) {
  mapping <- read.csv(opt$labels, sep = "\t", header = TRUE, colClasses = "character")
  lookup <- stats::setNames(mapping[["label"]], mapping[["beast_code"]])
  values <- as.character(beast@data[[trait_tag]])
  known <- !is.na(values) & values %in% names(lookup)
  values[known] <- unname(lookup[values[known]])
  beast@data[[trait_tag]] <- values
}

states <- as.character(beast[[trait_tag]])
unique_states <- sort(unique(states[!is.na(states)]))
n_states <- length(unique_states)
if (n_states > 9) {
  warning("More than 9 states. Switching to a larger palette.")
  colours <- colorRampPalette(brewer.pal(12, "Set3"))(n_states)
} else {
  colours <- colorRampPalette(brewer.pal(9, "Set1"))(n_states)
}
names(colours) <- unique_states
print(colours)

open_device <- function(file, ext, width, height) {
  if (ext == ".svg" && requireNamespace("svglite", quietly = TRUE)) {
    svglite::svglite(file, width = width, height = height)
  } else if (ext == ".svg") {
    grDevices::svg(file, width = width, height = height)
  } else if (ext == ".pdf") {
    grDevices::pdf(file, width = width, height = height)
  } else {
    grDevices::png(file, width = width, height = height, units = "in", res = 150)
  }
}

render <- function(plot, file, ext, width, height) {
  open_device(file, ext, width, height)
  print(plot)
  grDevices::dev.off()
  if (!file.exists(file)) {
    stop(
      sprintf(
        "the graphics device could not write %s; an SVG device needs a cairo-capable R",
        file
      ),
      call. = FALSE
    )
  }
}

p <- ggtree(beast, aes(color = .data[[trait_tag]]), size = 0.7, mrsd = mrsd) +
  scale_color_manual(values = colours, name = trait_tag) +
  theme_tree2() +
  theme(legend.position = c(.1, .8))

if (is.null(mrsd)) {
  p <- revts(p) + scale_x_continuous(labels = abs)
}

render(p, paste0(output_prefix, ext), ext, width, height)

prob_column <- paste0(trait_tag, ".prob")
if (prob_column %in% annotations) {
  p <- p + geom_nodelab(aes(label = round(as.numeric(.data[[prob_column]]), 2)), vjust = -.5, size = 3)
  render(p, paste0(output_prefix, ".", trait_tag, ".prob", ext), ext, width, height)
}
