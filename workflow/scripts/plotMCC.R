#!/usr/bin/env Rscript
#SNK --env ggtree

# Load the necessary libraries
library(ggplot2)
library(ggtree)
library(treeio)
library(optparse)
library(RColorBrewer)

# Set up the command line argument parser
option_list <- list(
  make_option(c("-i", "--input"),
    type = "character", default = NULL,
    help = "Path to the BEAST MCC tree file",
    metavar = "file"
  ),
  make_option(c("-o", "--output-prefix"),
    type = "character", default = "mcc.tree",
    help = "Output file for the tree plot (e.g., tree_plot.png)",
    metavar = "file"
  ),
  make_option(c("-m", "--mrsd"),
    type = "character",
    default = NULL,
    help = "Most recent sampling date (MRSD) for the tree", metavar = "date"
  ),
  make_option(c("-e", "--ext"),
    type = "character",
    default = ".svg",
    help = "Output file extension (e.g., .png, .pdf, .svg)", metavar = "ext"
  )
)

# Parse command line arguments
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Check if the file argument is provided
if (is.null(opt$input)) {
  stop(
    "No input file provided. Use --input to specify the BEAST MCC tree file.",
    call. = FALSE
  )
}

# Read in the BEAST tree output
beast <- read.beast(opt$input)

# Extract the file name without the extension
output_without_ext <- opt$`output-prefix`

# Extract the file extension
output_extension <- opt$ext
print(beast)
# Define the color palette for the grid
# Get unique locations
unique_locations <- sort(unique(beast[["location"]]))
n_locations <- length(unique_locations)

# Make sure you don’t exceed the maximum available in "Set1"
if (n_locations > 9) {
  warning("More than 9 unique locations. Switching to a larger palette.")
  colours <- colorRampPalette(brewer.pal(12, "Set3"))(n_locations)
} else {
  colours <- brewer.pal(n = n_locations, name = "Set1")
}

# Assign names
names(colours) <- unique_locations
print(colours)

# Create the plot
p <- ggtree(beast, aes(color = location), size = 0.7, mrsd = opt$mrsd) +
  scale_color_manual(values = colours) +
  theme_tree2() +
  theme(legend.position = c(.1, .8))

if (is.null(opt$mrsd)) {
  p <- revts(p) + scale_x_continuous(labels = abs)
}

ggsave(paste0(output_without_ext, output_extension), plot = p)

# p <- p +
#   geom_range(range = "height_0.95_HPD", color = "#3498db", alpha = 0.6, size = 2)

# ggsave(
#   paste0(output_without_ext, ".height_0.95_HPD", output_extension),
#   plot = p
# )

p <- p +
  geom_nodelab(aes(label = round(as.numeric(location.prob), 2)), vjust = -.5, size = 3)

ggsave(
  paste0(output_without_ext, ".location.prob", output_extension),
  plot = p
)
