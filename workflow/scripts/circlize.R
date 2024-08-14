# Load necessary libraries
library(circlize)
library(optparse)
library(RColorBrewer)

# Define command line options
option_list <- list(
    make_option(c("-l", "--log"),
        type = "character", default = NULL,
        help = "BEAST log file", metavar = "FILE"
    ),
    make_option(c("-o", "--output"),
        type = "character", default = NULL,
        help = "Output SVG file name", metavar = "OUTPUT"
    )
)

# Parse options
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Check if required options are provided
if (is.null(opt$log)) {
    stop("Input file must be provided using the -f or --file option.")
}

create_migration_matrix <- function(df) {
    # Filter columns that start with 'c_Location'
    location_cols <- colnames(df)[startsWith(colnames(df), "c_Location")]
    
    # Strip the last three characters '[1]' from each relevant column name
    cleaned_cols <- substr(location_cols, 1, nchar(location_cols) - 3)
    
    # Extract unique locations from the cleaned column names
    cols <- unique(gsub("c_Location\\.", "", cleaned_cols))
    
    locations <- unlist(strsplit(cols, "\\."))

    # take the first element of each part of the split
    locations <- sapply(locations, function(x) x[1])
    locations <- unique(locations)

    # Initialize a square matrix with zeros on the diagonal
    migration_matrix <- matrix(0,
        nrow = length(locations), ncol = length(locations),
        dimnames = list(locations, locations)
    )
    
    # Fill the migration matrix based on the cleaned columns
    for (i in seq_along(cleaned_cols)) {
        col <- cleaned_cols[i]
        
        # Extract origin and destination from the cleaned column name
        locs <- unlist(strsplit(gsub("c_Location\\.", "", col), "\\."))
        # Ensure locs has exactly two elements (origin and destination)
        if (length(locs) != 2) {
            stop(paste("Column name", location_cols[i], "does not split into two parts as expected"))
        }
        
        origin <- locs[1]
        destination <- locs[2]
        # Check if origin and destination exist in the locations
        if (!(origin %in% locations)) {
            stop(paste("Origin", origin, "not found in locations"))
        }
        if (!(destination %in% locations)) {
            stop(paste("Destination", destination, "not found in locations"))
        }
        # take the mean of the column
        average <- mean(df[[location_cols[i]]])
        # Populate the migration matrix using the original dataframe
        migration_matrix[origin, destination] <- average
    }
    return(migration_matrix)
}



# Read the input file
data <- read.csv(opt$log, sep = "\t", comment.char = "#", header = TRUE)
# Create the migration matrix
mean_migs <- create_migration_matrix(data)
print(mean_migs)
# Set up for the chord diagram
svg(opt$output, width = 6, height = 6)
arr.col <- data.frame(expand.grid(rownames(mean_migs), colnames(mean_migs)), rep("black", times = nrow(mean_migs) * ncol(mean_migs)))

# Example grid colors (customize as needed)
grid.col <- c(
    Central = "#2ecc71",
    Bua = "#3498db",
    Cakaudrove = "#9b59b6",
    Macuata = "#34495e",
    Taveuni = "#e74c3c"
)

grid.col <- brewer.pal(n = 9, name = "Set1")
names(grid.col) <- sort(rownames(mean_migs))
print(grid.col)
# Generate the chord diagram
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

# Add labels and titles
circos.trackPlotRegion(
    track.index = 1,
    bg.border = NA,
    panel.fun = function(x, y) {
        xlim <- get.cell.meta.data("xlim")
        sector.index <- get.cell.meta.data("sector.index")

        # Add names to the sector
        circos.text(
            x = mean(xlim),
            y = 2,
            labels = sector.index,
            facing = "bending",
            cex = 1
        )
    }
)

# Close the SVG device
dev.off()
