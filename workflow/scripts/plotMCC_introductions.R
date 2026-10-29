#!/usr/bin/env Rscript
# Annotated MCC tree for a discrete trait (derived from the analysis script used
# for the Australia introductions, generalised to the pipeline).
#
#   * tips are coloured by their sampled state,
#   * internal nodes are filled by the inferred state, with opacity from the
#     state posterior and size from the node posterior,
#   * nodes in a target state whose parent is in another state are marked as
#     introductions, with their tMRCA and 95% HPD interval.
#
# Everything dataset specific is a flag: the trait tag, the target states, the
# reference date.  Tip states come from the tree annotations (the logTree writes
# them) and fall back to the trait table; no external metadata sheet, and no
# ggrepel/readr, so the rule runs in the existing R environment.
#
# treeannotator (BEAST 1.10) annotates node heights but not their intervals, so
# the HPD of an introduction date is computed from the posterior trees: the
# clade below the introduction node is located in every tree and its MRCA height
# is collected.  Pass --posterior to enable it; without it the node is marked
# with a point instead of a bar.
suppressPackageStartupMessages({
  library(ape)
  library(dplyr)
  library(ggplot2)
  library(ggtree)
  library(purrr)
  library(scales)
  library(tibble)
  library(treeio)
})

parse_args <- function(args) {
  parsed <- list()
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) stop(sprintf("unexpected argument %s", key), call. = FALSE)
    parsed[[sub("^--", "", key)]] <- args[[i + 1]]
    i <- i + 2
  }
  parsed
}

opt <- parse_args(commandArgs(trailingOnly = TRUE))
if (is.null(opt$tree)) stop("--tree is required", call. = FALSE)
if (is.null(opt$tag)) stop("--tag is required", call. = FALSE)
if (is.null(opt[["output-prefix"]])) stop("--output-prefix is required", call. = FALSE)
if (is.null(opt$mrsd)) stop("--mrsd is required for the calendar axis", call. = FALSE)

tree_file <- opt$tree
trait_tag <- opt$tag
output_prefix <- opt[["output-prefix"]]
trait_file <- opt$traits
posterior_file <- opt$posterior
mrsd <- as.Date(opt$mrsd)
min_prob <- if (is.null(opt[["min-prob"]])) 0.7 else as.numeric(opt[["min-prob"]])
#: only the earliest this-many entries are labelled; the table keeps them all
max_labels <- if (is.null(opt[["max-labels"]])) 15L else as.integer(opt[["max-labels"]])
target_states <- if (is.null(opt$states) || !nzchar(opt$states)) {
  character(0)
} else {
  trimws(strsplit(opt$states, ",")[[1]])
}

decimal_year <- function(x) {
  x <- as.Date(x)
  year_start <- as.Date(format(x, "%Y-01-01"))
  next_year_start <- as.Date(sprintf("%d-01-01", as.integer(format(x, "%Y")) + 1L))
  as.integer(format(x, "%Y")) +
    as.numeric(x - year_start) / as.numeric(next_year_start - year_start)
}

format_year <- function(x) sprintf("%.2f", x)

# Shortest interval holding `level` of the values (the usual HPD).
hpd_interval <- function(values, level = 0.95) {
  values <- sort(values[is.finite(values)])
  n <- length(values)
  if (n < 2) return(c(NA_real_, NA_real_))
  k <- max(1, floor(level * n))
  widths <- values[(k + 1):n] - values[1:(n - k)]
  start <- which.min(widths)
  c(values[start], values[start + k])
}

tree <- read.beast(tree_file)
mcc_tree <- as.phylo(tree)
tree_tbl <- as_tibble(tree)
mrsd_year <- decimal_year(mrsd)
prob_column <- paste0(trait_tag, ".prob")

column_or_na <- function(tbl, column, mode = as.character) {
  if (column %in% names(tbl)) mode(tbl[[column]]) else rep(mode(NA), nrow(tbl))
}

annotate <- function(tbl) {
  tbl$height <- if ("height" %in% names(tbl)) as.numeric(tbl$height) else NA_real_
  tbl$posterior <- if ("posterior" %in% names(tbl)) as.numeric(tbl$posterior) else NA_real_
  tbl$state <- column_or_na(tbl, trait_tag)
  tbl$state_prob <- if (prob_column %in% names(tbl)) as.numeric(tbl[[prob_column]]) else NA_real_
  tbl
}

tree_tbl <- annotate(tree_tbl)

# Tips: the tree annotation is authoritative, the trait table is the fallback.
if (!is.null(trait_file) && file.exists(trait_file)) {
  trait_table <- read.delim(trait_file, check.names = FALSE, stringsAsFactors = FALSE)
  tree_tbl <- tree_tbl |>
    left_join(
      trait_table |> transmute(label = taxon, state_from_table = state),
      by = "label"
    ) |>
    mutate(state = coalesce(state, state_from_table)) |>
    select(-state_from_table)
}

tip_tbl <- tree_tbl |> filter(!is.na(label)) |> mutate(state = ifelse(is.na(state), "unknown", state))
node_tbl <- tree_tbl |> filter(is.na(label))

# Introductions: a node in a target state whose parent is in a different state.
introductions <- node_tbl[0, ]
if (length(target_states) && nrow(node_tbl)) {
  parent_state <- tree_tbl$state[match(tree_tbl$parent, tree_tbl$node)]
  introductions <- tree_tbl |>
    mutate(parent_state = parent_state) |>
    # an introduction is an internal node, so it has a parent and no label
    filter(is.na(label), !is.na(parent), state %in% target_states, !(parent_state %in% target_states)) |>
    mutate(tmrca = mrsd_year - height)
}

# HPD of each introduction date from the posterior trees: the clade under the
# node is the same clade in every tree, so its MRCA height is comparable.
if (nrow(introductions) && !is.null(posterior_file) && file.exists(posterior_file)) {
  posterior <- read.nexus(posterior_file)
  if (!inherits(posterior, "multiPhylo")) posterior <- structure(list(posterior), class = "multiPhylo")
  # prop.part() lists only the internal nodes, in node order, so internal node
  # number `node` (Ntip + 1 .. Nnode) is element `node - Ntip`.
  partitions <- prop.part(mcc_tree)
  n_tips <- Ntip(mcc_tree)
  heights_by_node <- function(node) {
    # tip *labels*: numbering differs between the MCC and the posterior trees
    tips <- mcc_tree$tip.label[partitions[[node - n_tips]]]
    # vapply, not purrr::map_dbl: mapping over a multiPhylo returns NA for every
    # tree (the class carries attributes purrr's proxy mishandles)
    vapply(
      posterior,
      function(one) {
        mrca <- tryCatch(getMRCA(one, tips), error = function(e) NA_integer_)
        if (is.null(mrca) || is.na(mrca)) return(NA_real_)
        depth <- node.depth.edgelength(one)
        tip_depth <- max(depth[seq_len(Ntip(one))])
        mrsd_year - (tip_depth - depth[mrca])
      },
      numeric(1)
    )
  }
  intervals <- lapply(introductions$node, function(node) hpd_interval(heights_by_node(node)))
  introductions$hpd_lower <- vapply(intervals, `[[`, numeric(1), 1)
  introductions$hpd_upper <- vapply(intervals, `[[`, numeric(1), 2)
} else {
  introductions$hpd_lower <- NA_real_
  introductions$hpd_upper <- NA_real_
}
introductions <- introductions |> arrange(tmrca) |>
  mutate(introduction = paste("Introduction", row_number()))

base_plot <- ggtree(tree, mrsd = mrsd, size = 0.25, color = "grey55")
plot_tbl <- annotate(as_tibble(base_plot$data)) |>
  left_join(select(tip_tbl, label, state_tip = state), by = "label") |>
  mutate(state = coalesce(state_tip, state)) |>
  select(-state_tip)

tip_plot_tbl <- plot_tbl |> filter(!is.na(label)) |> mutate(state = ifelse(is.na(state), "unknown", state))
node_plot_tbl <- plot_tbl |>
  filter(is.na(label), !is.na(state), !is.na(posterior)) |>
  mutate(node_alpha = if_else(is.na(state_prob) | state_prob < min_prob, 0.15, 1.0))

intro_plot_tbl <- plot_tbl |>
  filter(node %in% introductions$node) |>
  select(node, x, y) |>
  left_join(
    introductions |> select(node, introduction, tmrca, hpd_lower, hpd_upper),
    by = "node"
  ) |>
  mutate(
    bar_lower = x + (hpd_lower - tmrca),
    bar_upper = x + (hpd_upper - tmrca),
    tmrca_label = format_year(x)
  ) |>
  arrange(x)

states <- sort(unique(c(tip_plot_tbl$state, node_plot_tbl$state)))
state_cols <- setNames(hue_pal(l = 65, c = 100)(length(states)), states)
state_cols[["unknown"]] <- "grey70"

tree_plot <- base_plot +
  geom_tippoint(
    data = tip_plot_tbl,
    mapping = aes(x = x, y = y, color = state),
    size = 2.2,
    alpha = 0.95,
    inherit.aes = FALSE
  ) +
  geom_point(
    data = node_plot_tbl,
    mapping = aes(x = x, y = y, fill = state, size = posterior, alpha = node_alpha),
    shape = 21,
    color = "grey20",
    stroke = 0.3,
    inherit.aes = FALSE
  ) +
  scale_color_manual(values = state_cols, name = trait_tag, na.value = "grey70") +
  scale_fill_manual(values = state_cols, na.value = "transparent", guide = "none") +
  scale_alpha_identity(guide = "none") +
  scale_size_continuous(name = "Posterior", range = c(0.2, 4), limits = c(0, 1)) +
  scale_x_continuous(
    name = "Year",
    breaks = pretty_breaks(n = 7),
    expand = expansion(mult = c(0.01, 0.18))
  ) +
  theme_tree2() +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.title.x = element_text(size = 12),
    axis.text.x = element_text(size = 9, angle = 45, hjust = 1),
    legend.title = element_text(size = 11),
    legend.text = element_text(size = 9),
    panel.grid.major.x = element_line(color = "grey92", linewidth = 0.3),
    panel.grid.minor.x = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA)
  ) +
  guides(color = guide_legend(override.aes = list(size = 3, alpha = 1)))

if (nrow(intro_plot_tbl)) {
  # a bar only where the interval has width; otherwise the node is the estimate
  wide <- !is.na(intro_plot_tbl$hpd_lower) &
    (intro_plot_tbl$hpd_upper - intro_plot_tbl$hpd_lower) > 1e-3
  if (any(wide)) {
    tree_plot <- tree_plot +
      geom_segment(
        data = intro_plot_tbl[wide, ],
        mapping = aes(x = bar_lower, xend = bar_upper, y = y, yend = y),
        linewidth = 1.2, color = "#7A3DB8", alpha = 0.8,
        inherit.aes = FALSE
      )
  }
  tree_plot <- tree_plot +
    geom_point(
      data = intro_plot_tbl,
      mapping = aes(x = x, y = y),
      shape = 23, size = 2.4, fill = "#7A3DB8", color = "white", stroke = 0.4,
      inherit.aes = FALSE
    ) +
    # label the earliest entries only: a state that is entered hundreds of times
    # would otherwise cover the tree with text (the CSV keeps every row)
    geom_label(
      data = head(intro_plot_tbl, max_labels),
      mapping = aes(x = x, y = y, label = tmrca_label),
      color = "#7A3DB8", fill = "white", size = 2.6,
      label.padding = unit(0.12, "lines"), nudge_y = 1.5,
      inherit.aes = FALSE
    )
}

summary_tbl <- if (nrow(introductions)) {
  introductions |>
    transmute(
      introduction, node, posterior,
      inferred_state = state, parent_state,
      tmrca, hpd_lower, hpd_upper
    )
} else {
  tibble(
    introduction = character(), node = integer(), posterior = numeric(),
    inferred_state = character(), parent_state = character(),
    tmrca = numeric(), hpd_lower = numeric(), hpd_upper = numeric()
  )
}

write.csv(summary_tbl, paste0(output_prefix, ".csv"), row.names = FALSE)
ggsave(paste0(output_prefix, ".png"), tree_plot, width = 12, height = 10, dpi = 300, bg = "white")
ggsave(paste0(output_prefix, ".pdf"), tree_plot, width = 12, height = 10, bg = "white")

message("Wrote figure to ", paste0(output_prefix, ".png"))
message("Wrote summary to ", paste0(output_prefix, ".csv"), " (", nrow(summary_tbl), " rows)")
