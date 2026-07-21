#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
FEATURE_DIR <- if (length(args) >= 1) {
  normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
} else {
  normalizePath(file.path(getwd(), "unified_atac_features"), winslash = "/", mustWork = TRUE)
}
OUT_DIR <- if (length(args) >= 2) {
  normalizePath(args[[2]], winslash = "/", mustWork = FALSE)
} else {
  FEATURE_DIR
}
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

model <- readRDS(file.path(FEATURE_DIR, "lsi_model.rds"))
v <- as.matrix(model$svd_v)
idf <- as.numeric(model$idf)
d <- as.numeric(model$svd_d)

stopifnot(length(idf) == nrow(v))

out <- file.path(OUT_DIR, "lsi_projection_model.bin")
con <- file(out, open = "wb")
on.exit(close(con), add = TRUE)
writeBin(as.integer(c(length(idf), nrow(v), ncol(v), length(d))), con, size = 4, endian = "little")
writeBin(idf, con, size = 8, endian = "little")
writeBin(as.numeric(v), con, size = 8, endian = "little")
writeBin(d, con, size = 8, endian = "little")
writeBin(as.numeric(model$scale), con, size = 8, endian = "little")

meta <- data.frame(
  item = c("n_peaks", "n_svd_rows", "n_lsi_dims", "n_singular_values", "tfidf_scale"),
  value = c(length(idf), nrow(v), ncol(v), length(d), as.numeric(model$scale))
)
write.table(meta, file.path(OUT_DIR, "lsi_projection_model.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
cat("Wrote", out, "\n")
