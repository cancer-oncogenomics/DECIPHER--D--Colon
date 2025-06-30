#!/dssg/softwares/R/R-3.6.0/bin/Rscript

# 20211124:  output frag2023.arm no scale results.
# 20211206:  add bin name and range to 100kb bins.
# 20230112:  fix error in BED, fix NA in LOESS-based GC correction, speed up LOESS

suppressPackageStartupMessages(library(optparse))
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(reshape2))
suppressPackageStartupMessages(library(BiocGenerics))
suppressPackageStartupMessages(library(GenomicRanges))
suppressPackageStartupMessages(library(stats4))
suppressPackageStartupMessages(library(S4Vectors))
suppressPackageStartupMessages(library(IRanges))
suppressPackageStartupMessages(library(GenomeInfoDb))

option_list<-list(make_option(c("-s", "--sample"), type="character", default=NULL, 
                              help="sample name", metavar="character"),
                  make_option(c("--counts_by_width"), type="character", default=NULL, metavar="character"),
                  make_option(c("--counts"), type="character", default=NULL, metavar="character"),
                  make_option(c("--bingc"), type="character", default=NULL, metavar="character"),
                  make_option(c("--range_start"), type="integer", default=100, 
                              help="range start. default: 100", metavar="integer"),
                  make_option(c("--range_end"), type="integer", default=400, 
                              help="range end. default: 400", metavar="integer"),
                  make_option(c("--step"), type="integer", default=5, 
                              help="step of fragment length between range, default 5. ", metavar="integer"), 
                  make_option(c("--binsize"), type="double", default=5, 
                              help="bin size in units of M, the minimum unit: 0.1M(100k), default 5. ", metavar="double"),
                  make_option(c("--ab"), type="character", default=NULL,
                              help="AB.rds", metavar="character"),
                  make_option(c("--f_output"), type="character", default=NULL,
                              help="f_output", metavar="character")
)

opt_parser<-OptionParser(option_list=option_list)
opt<-parse_args(opt_parser)

sample = opt$sample
#dir.create(py_cmd,recursive = T)
range_start = opt$range_start
range_end = opt$range_end
ab_rds = opt$ab
f_output = opt$f_output

message(Sys.time(), " - Program Start")

###### =====> 01 read_galp <=====

# GC 校正
gc.correct <- function(coverage, bias) {
    # coverage: 与每个 AB 区间重叠的片段数量
    # bias：与每个 AB 区间重叠的片段平均 GC 含量
    # i: 从小到大以 0.001 为步长建平均 GC 含量的梯度，
    i <- seq(min(bias, na.rm=TRUE), max(bias, na.rm=TRUE), by = 0.001)
    # 为局部多项式拟合补齐最后一个端点（即 GC 最大值的那一个点）
    j <- c(i, max(bias, na.rm=TRUE))
    # converage.trend: 片段数量 ～ 平均GC含量，进行局部多项式回归（非参估计）
    # fit.control: LOESS 加速的选项，推荐在样本点较多（> 1000）时开启
    fit.control <- loess.control(
        surface = "interpolate",
        statistics = "approximate",
        trace.hat = "approximate")
    coverage.trend <- loess(coverage ~ bias, control = fit.control)
    # coverage.model：在 0.001 步长的精度上再次进行局部多项式回归（二次平滑）
    coverage.model <- loess(predict(coverage.trend, j) ~ j)
    # 用二次平滑的模型预测在各平均 GC 含量条件下与 AB 区间重叠的片段数量
    coverage.pred <- predict(coverage.model, bias)
    # GC校正后的重叠片段数量 = 原重叠片段数 - 二次平滑预测值 + 原重叠片段数中位值
    coverage.corrected <- coverage - coverage.pred + median(coverage)
}

AB = readRDS(ab_rds)
message(Sys.time(), " - AB loaded")

# calculate fragment period
counts = read.csv(opt$counts_by_width, header = FALSE)
total_counts = read.csv(opt$counts, header = FALSE)[, 1]
bingc = read.csv(opt$bingc, header = FALSE)[, 1]
count_gc = data.frame(total = total_counts)
message(Sys.time(), " - counts files loaded")
frag_period_list=seq(range_start, range_end ,by = opt$step)
for (i in 2:length(frag_period_list)){
  feature_name = paste("frag",frag_period_list[i-1],frag_period_list[i]-1, sep = ".")
  count_gc[,feature_name] = rowSums(counts[,(frag_period_list[i-1] - range_start + 3):(frag_period_list[i] - range_start + 2)])
  count_gc[,feature_name] = gc.correct(count_gc[,feature_name], bingc)
  message(Sys.time(), " - GC correction for ", feature_name)
}

AB$frag.gc <- bingc

for(i in 1:ncol(count_gc)) elementMetadata(AB)[,colnames(count_gc)[i]] <- count_gc[,i]

###### =====> 04 bins in unit of Mb <=====
#### 04.1 binsize ####

tib.list = list(sample = as_tibble(AB))
names(tib.list) <- sample
tib.list <- map2(tib.list, names(tib.list), ~ mutate(.x, id = .y)) %>%
  bind_rows() %>% dplyr::select(id, everything())
tib.list <- tib.list %>% dplyr::select(-matches("X"))

df.fr2 = tib.list
armlevels <- c("1p","1q","2p","2q","3p","3q","4p","4q","5p","5q","6p","6q",
               "7p","7q","8p","8q", "9p", "9q","10p","10q","11p","11q","12p",
               "12q","13q","14q","15q","16p","16q","17p","17q","18p","18q",
               "19p", "19q","20p","20q","21q","22q")
df.fr2$arm <- factor(df.fr2$arm, levels=armlevels)

## combine adjacent 100kb bins to form 5mb bins. We count starting from
## the telomeric end and remove the bin closest to the centromere if it is
## smaller than 5mb.

binsize = opt$binsize * 10

# df.fr2 <- df.fr2 %>% group_by(id, arm) %>%
#   mutate(combine = ifelse(grepl("p", arm), ceiling((1:length(arm))/binsize),
#                           ceiling(rev((1:length(arm))/binsize) )))
message(Sys.time(), " - bins in unit of Mb")

#### 04.2 fragment step ####
df.fr4 = data.frame(id=character(), seqnames = character(), arm = character(), nfrag = double(), frag_seq = character(), stringsAsFactors = F)
for (i in 2:length(colnames(count_gc))){
  tmp = df.fr2[,c("id", "seqnames", "arm", colnames(count_gc)[i])] 
  colnames(tmp)[4] = "nfrag"
  tmp.df = tmp  %>% group_by(id, seqnames, arm) %>%
    summarize(nfrag = sum(nfrag), .groups = "drop")
  tmp.df$frag_seq = colnames(count_gc)[i]
  df.fr4 = rbind(df.fr4, data.frame(tmp.df))
}
message(Sys.time(), " - fragment step")

df.fr4$feature = paste("FragArm2023", df.fr4$seqnames ,df.fr4$arm, df.fr4$frag_seq, sep = ".")

df.fr4$nfrag_scale = scale(df.fr4$nfrag)
features.window = reshape2::dcast(df.fr4, id ~ feature, value.var = "nfrag_scale")
features.window = features.window[df.fr4$feature]
features.columns = c("SampleID", colnames(features.window))
features.window$SampleID = sample
features.window = features.window[, features.columns]
# rownames(features.window) = sample
# names(dimnames(features.window)) = c("SampleID", "")
write.csv(features.window, file = f_output , row.names = F, quote = FALSE)
message(Sys.time(), " - done")
