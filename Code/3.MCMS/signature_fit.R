suppressPackageStartupMessages(library(MutationalPatterns))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(optparse))

option_list <- list(
    make_option(c("--sample"), type = "character", default = NULL,
                help = "sample id", metavar = "character"),
    make_option(c("--context"), type = "character", default = NULL,
                help = "mutational context, SNP_in, long format",
                metavar = "character"),
    make_option(c("--reference"), type = "character", default = NULL,
                help = "reference mutational signatures",
                metavar = "character"),
    make_option(c("--output"), type = "character", default = NULL,
                help = "output file, regular refitting", metavar = "character")
)
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# how to get signatures from the MutatioanlPatterns package:
#     signatures_v32 <- get_known_signatures(
#         muttype = "snv", source = "COSMIC", sig_type = "reference",
#         genome = "GRCh37", incl_poss_artifacts = TRUE)

df_mut <- readr::read_tsv(opt$context, show_col_types = FALSE) |>
    tibble::column_to_rownames("context_3nt")
mut_mat <- as.matrix(df_mut |> select(-`context`))

signatures <- read.table(opt$reference, sep = "\t", header = TRUE)
signatures <- signatures |> select(-Type)
signatures <- t(t(signatures))
artefacts <- c(34, 50, seq(52, 67), 79)

# construct a dummy list of signature fitting results
null_contrib <- as.matrix(rep(0.0, ncol(signatures)))
colnames(null_contrib) <- opt$sample
rownames(null_contrib) <- colnames(signatures)
null_recon <- as.matrix(rep(0.0, nrow(mut_mat)))
colnames(null_recon) <- opt$sample
fit_res_null <- list(
    contribution = null_contrib,
    reconstructed = null_recon
)

mat_to_MC <- function(mat, sample_id, col_names) {
    df <- as.data.frame(t(as.data.frame(mat)))
    colnames(df) <- col_names
    # rearrange by original MC order
    df <- df[, sort(colnames(df))]
    # add SampleID at 1st column
    df <- cbind(data.frame(SampleID = sample_id), df)
    df
}

if (sum(mut_mat) > 0) {
    fit_res <- fit_to_signatures(mut_mat, signatures)
} else {
    fit_res <- fit_res_null
}
# export MS signature contribution fit
contrib <- as.data.frame(t(as.data.frame(fit_res$contribution)))
colnames(contrib) <- lapply(colnames(contrib), function(x) paste0("MS.", x))
contrib <- cbind(data.frame(SampleID = opt$sample), contrib)
write.csv(contrib, opt$output, quote = FALSE, row.names = FALSE)
