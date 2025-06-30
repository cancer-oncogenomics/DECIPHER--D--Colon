# Summarize mutational contexts from variants annotation with GC correction

suppressPackageStartupMessages(library(plyr))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(stringr))

suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(IRanges))
suppressPackageStartupMessages(library(GenomicRanges))
suppressPackageStartupMessages(library(MutationalPatterns))

ref_genome <- "BSgenome.Hsapiens.1000genomes.hs37d5"
suppressPackageStartupMessages(library(ref_genome, character.only = TRUE))

# p2 --------------------
parse_mutations <- function(file_name, prefix, sam, size_selection = NA) {

  message(paste0("files to run on :\n", file_name))

  Sys.sleep(2)

  # load ANNOVAR annotated mutations
  colnames_used <- c(
    "Chr", "Start", "End", "Ref", "Alt",
    "Func.refGene", "Gene.refGene", "GeneDetail.refGene",
    "ExonicFunc.refGene", "AAChange.refGene",
    "cytoBand", "avsnp151", "Otherinfo")
  anno <- read_tsv(file_name, col_names = colnames_used, show_col_types = FALSE)

  # filtering and tidying
  wide <- anno |>
    # get rid of comment lines
    filter(row_number() > 1 & str_detect(Chr, "#", negate = TRUE)) |>
    # patch seqnames to UCSC style, fill in sample ID and total bp count
    mutate(
      # Chr = paste0("chr", Chr),  # not applicable to hs37d5
      SampleID = gsub(file_name,
        pattern = "09_annotation/(.*).hg19_multianno.*", replacement = "\\1"),
      bp = as.numeric(str_split_i(Otherinfo, " ", i = 3))
    ) |>
    # keep SNV only
    filter(str_length(Ref) == 1 & str_length(Alt) == 1) |>
    mutate(mut_class = paste0(Ref, ">", Alt)) |>
    # extract SNV depth info
    # DP4 = Number of high-quality bases of:
    #   - ref-forward
    #   - ref-reverse
    #   - alt-forward
    #   - alt-reverse
    # DP4 = 0,0,1,1 means the SNV is confirmed on both strands
    mutate(
      DP4 = gsub(
        Otherinfo, pattern = ".*DP4=(.,.,.,.).*",
        replacement = "\\1", perl = TRUE)) |>
    filter(str_detect(DP4,"comment:", negate = TRUE)) |>
    # fill in SNV informations
    mutate(
      # calculate SNV reads based on DP4 (`filed 3` + `field 4`)
      mut_reads = (
        as.numeric(str_split_i(DP4, ",", i = 3)) +
        as.numeric(str_split_i(DP4, ",", i = 4))
      )
    )

  # dbsnp filter
  if (grepl(file_name, pattern = "SNP_out") == TRUE) {
    message("removing SNPs based on avsnp151")
    wide <- wide |> filter(avsnp151 == ".")
  }

  message("annotating context with 6 mutation classes, 3nt context")
  wide <- add_context(wide, ref_genome, "hs37d5", "Ensembl")

  message(paste0("saving to: ", "10_context/", prefix, ".raw.rds"))
  saveRDS(wide, paste0("10_context/", prefix, ".raw.rds"))
  write.table(
    wide, paste0("10_context/", prefix, ".raw.tsv"),
    sep = "\t", quote = FALSE, row.names = FALSE
  )

  # generate long output
  long <- plyr::ddply(
    wide, c("SampleID", "context", "mut_reads", "DP4"),
    function(x) {
      data.frame(n_mut = nrow(x))
    }
  )
  saveRDS(long, paste0("10_context/", prefix, ".long.rds"))
  write.table(
    long, paste0("10_context/", prefix, ".long.tsv"),
    sep = "\t", quote = FALSE, row.names = FALSE
  )

  # generate a size annotated 0011 df
  message("starting size ann")
  wide_size_0011 <- annotate_size(sam = sam, raw = wide)

  if (!is.na(size_selection)) {
    message(paste("performing size selection, threshold = ", size_selection))

    wide_size_0011_short <- filter(wide_size_0011, length < size_selection)
    wide_size_0011_long <- filter(wide_size_0011, length >= size_selection)

    # generate output [short fragments]
    message("saving short fragment long.rds")

    # GC correct the size.rds
    message("starting gc normalisation - short fragments")
    gc_normalization(wide_size_0011_short,
                     paste0(prefix, ".size_selected_below_", size_selection))

    # GC correct the size.rds
    message("starting gc normalisation")
    gc_normalization(wide_size_0011_long,
                     paste0(prefix, ".size_selected_above_", size_selection))
  } else {
    message("no size selection")
    message(paste0("saving to: ", "10_context/", prefix, ".raw.size.rds"))
    saveRDS(wide_size_0011, paste0("10_context/", prefix, ".raw.size.rds"))
    write.table(
      wide_size_0011, paste0("10_context/", prefix, ".raw.size.tsv"),
      sep = "\t", quote = FALSE, row.names = FALSE
    )
  }

  # GC correct the size.rds
  message("starting gc normalisation")
  gc_normalization(wide_size_0011, prefix)


  return(NULL)

}

# annotate mutations with trinucleotide contexts
add_context <- function(mut, genome, genome_name, seq_style) {
  message("annotating with mutation context and converting from 12 to 6 classes")

  mut_granges <- makeGRangesFromDataFrame(
    mut, seqnames.field = "Chr", start.field = "Start", end.field = "End",
    keep.extra.columns = TRUE
  )
  GenomeInfoDb::genome(mut_granges) <- genome_name
  GenomeInfoDb::seqlevelsStyle(mut_granges) <- seq_style

  mut_granges$trinucleotide <- mut_context(mut_granges, genome)

  mut <- as.data.frame(mut_granges)

  # convert from 12 to 6 mut_classes then generate context column
  pyr_mut_classes <- c("C>A", "C>G", "C>T", "T>A", "T>C", "T>G")
  wide_a <- mut |> filter(mut_class %in% pyr_mut_classes)
  wide_b <- mut |>
    filter(!(mut_class %in% pyr_mut_classes)) |>
    mutate(
      mut_class = chartr("ATGC", "TACG", mut_class),
      trinucleotide = IRanges::reverse(chartr("ATGC", "TACG", trinucleotide))
    )
  wide <- as_tibble(
    rbind(wide_a, wide_b) |>
      mutate(
        context = str_c(
          str_sub(trinucleotide, 1, 1),
          "[", mut_class, "]",
          str_sub(trinucleotide, 3, 3)
        )
      )
  )

  return(wide)
}

annotate_size <- function(sam, raw) {
  message("annotating raw df with size info")
  message("generating a size annotated 0011 df")

  # take only 15 cols
  sam <- read.table(sam, fill = TRUE, row.names = NULL, header = FALSE)
  print(head(sam))

  # only take high quality mapping reads
  sam <- filter(sam, V5 == 60)
  print(head(sam))

  # parse
  message("parsing sam")

  # dynamically determine colnames
  # use the majority cigar string so that the colnames are accurate
  cigars <- data.frame(table(sam$V6))
  majority_cigar <- as.character(
    cigars[cigars$Freq == max(cigars$Freq), "Var1"])
  message(majority_cigar)
  sam_head <- filter(sam, V6 == majority_cigar & V13 != "")[1, ]
  message("SAM header")
  print(sam_head)

  sam_t <-  t(sam_head)
  message("transposed SAM header")
  print(sam_t)

  message("check if column names are set correctly for the SAM")
  sam_t[1:11, ] <- c(
    "name", "flag", "chr", "pos", "MQ.1", "CIGAR", "mate", "mate_start",
    "length", "sequence", "quality_scores")
  sam_t <- gsub(sam_t, pattern = "(..):.*", replacement = "\\1")
  print(sam_t)

  # assign new colnames
  print(paste("new colnames:", as.vector(sam_t)))
  colnames(sam) <- sam_t
  print(head(sam))

  # deduplicate and tidy
  duplicated <- sam$name[duplicated(sam$name)]

  sam$MD <- gsub(sam$MD, pattern = "MD:Z:", replacement = "")
  sam <- sam[sam$NM == "NM:i:1", ]

  # here, the distance to the mutation is obtained from the CIGAR to calculate
  # the uniq_pos of the mutation, which should correspond with the pileup
  sam$dist <- as.numeric(
    gsub(sam$MD, pattern = "(.*)([ACGT])(.*)", replacement = "\\1"))
  sam$alt <- gsub(sam$MD, pattern = "(.*)([ACGT])(.*)", replacement = "\\2")
  sam <- sam[!is.na(sam$dist), ]
  sam$uniq_pos <- paste0(
    "chr", sam$chr, ":",
    as.numeric(as.character(sam$pos)) + as.numeric(as.character(sam$dist))
  )
  print(head(sam))

  message("filtering sam")
  # remove blank columns
  sam <- sam[, !(colnames(sam) == "")]

  # filter
  fr_loci <- filter(data.frame(table(sam$uniq_pos)), Freq == 2)[, 1]
  sam$FR_support <- sam$uniq_pos %in% fr_loci
  sam$XS <- gsub(sam$XS, pattern = "XS:i:", replacement = "")
  sam$XS <- as.numeric(sam$XS)

  sam$AS <- gsub(sam$AS, pattern = "AS:i:", replacement = "")
  sam$AS <- as.numeric(sam$AS)
  sam <- filter(sam, length > 0)
  print(head(sam))

  raw$uniq_pos <- paste0(raw$seqnames, ":", raw$start)

  message("filtering DP4 = 0011")
  raw_0011 <- filter(raw, DP4 == "0,0,1,1")
  # !!! sam file using UCSC style seqname, i.e. "chr1"
  raw_0011$uniq_pos <- paste0("chr", raw_0011$uniq_pos)
  print(head(raw_0011$uniq_pos))
  print(head(sam$uniq_pos))

  message("merging raw df with lengths + quality metrics from SAM")

  raw_0011 <- left_join(
    raw_0011, sam[, c("uniq_pos", "length", "sequence", "dist", "XS", "AS")],
    by = "uniq_pos")
  print(head(raw_0011))

  print(raw_0011$sequence[1:5])

  message("returning annotated raw file")
  return(raw_0011)
}

# GC normalization
gc_normalization <- function(curr_size_0011, prefix) {
  # get normalisation data---------------------------
  # note that this will get GC data from all filtered.txt files
  suppressPackageStartupMessages(library(purrr))
  suppressPackageStartupMessages(library(data.table))
  suppressPackageStartupMessages(library(ggplot2))
  suppressPackageStartupMessages(library(seqinr))

  message("reading in GC bias files")
  read_plus <- function(flnm) {
      fread(flnm, header = TRUE, showProgress = TRUE, skip = 6) %>%
      mutate(file_name = flnm) %>%
      as.data.table()
  }
  pattern <- list.files(
    "05_collect_gc", pattern = "gc_bias_metrics.txt", full.names = TRUE)

  output <- pattern %>% map_df(~read_plus(.))
  output$SampleID <- gsub(
    output$file_name, pattern = "05_collect_gc/(.*).gc_bias_metrics.*",
    replacement = "\\1")

  message("applying the smoother using ggplot")
  p <- ggplot(output, aes(x = GC, y = NORMALIZED_COVERAGE)) + geom_smooth()
  fit <- ggplot_build(p)$data[[1]]

  predictions <- data.frame(predict(loess(y ~ x, fit, span = 0.1), seq(0, 100)))

  predictions$GC <- seq(0, 100)
  colnames(predictions)[1] <- "fit"

  output1 <- left_join(output, predictions, by = "GC")
  print(output1$fit)
  print(output1$NORMALIZED_COVERAGE)

  # generate a scaling vector for each sample
  message("generating a scaling vector for each factor")
  output1$fit <- pmax(output1$fit, 0)
  output1$scale_factor <- output1$fit / output1$NORMALIZED_COVERAGE
  output1[output1$READ_STARTS == 0, "scale_factor"] <- NA
  print(output1$fit)
  print(output1$NORMALIZED_COVERAGE)
  print(output1$scale_factor)
  print(unique(output1$SampleID))

  #annotate the mutations df with GC content
  message("GC annotation using seqinr")
  curr_size_0011$sequence1 <- as.character(curr_size_0011$sequence)
  curr_size_0011$GC <- unlist(lapply(lapply(curr_size_0011$sequence1, s2c), GC))
  curr_size_0011$GC <- round(curr_size_0011$GC * 100)
  message("check if barcode is correct")
  message(unique(curr_size_0011$SampleID))
  curr_size_0011$SampleID <- gsub(
    curr_size_0011$SampleID, pattern = "(.*).SNP_.*",
    replacement = "\\1")
  message(unique(curr_size_0011$SampleID))

  # merge
  message("(2nd merge) merging scaling vector with curr_size_0011 df")
  test <- left_join(
    curr_size_0011,
    output1[, c("SampleID", "scale_factor", "GC")],
    by = c("SampleID", "GC"))
  print(as_tibble(test))

  message("saving GC normalized size.rds")
  saveRDS(test, paste0("10_context/", prefix, ".raw.size.GC_normalized.rds"))
  write.table(
    test, paste0("10_context/", prefix, ".raw.size.GC_normalized.tsv"),
    sep = "\t", quote = FALSE, row.names = FALSE
  )

  # generate long output
  long <- plyr::ddply(
    test, c("SampleID", "context", "mut_reads", "DP4"),
    function(x) {
      # NAs can occur in this df due to supplementary alignments
      # - OK to exclude them
      data.frame(n_mut.normalized = round(sum(x$scale_factor, na.rm = TRUE)))
    })

  message("saving GC normalized long.rds")
  saveRDS(long, paste0("10_context/", prefix, ".long.GC_normalized.rds"))
  write.table(
    long, paste0("10_context/", prefix, ".long.GC_normalized.tsv"),
    sep = "\t", quote = FALSE, row.names = FALSE
  )

  return(NULL)
}

## set variables
args <- commandArgs(TRUE)
file_name <- args[1]
prefix <- args[2]
sam <- args[3]
size_selection <- args[4]

# echo
message(paste("file_name:", file_name))
message(paste("prefix:", prefix))
message(paste("sam path:", sam))

parse_mutations(file_name, prefix, sam, size_selection)
