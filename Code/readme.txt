The study employed three key features - CNV, ARM-FSD, and MCMS. Below are the detailed analytical commands for these features along with the specific model training code.
Note: The larger ANNOVAR database and genome reference files are not provided here and need to be downloaded from public databases.

## CNV
The CNV feature analysis requires installation of readCounter and ichorCNA.
The detailed analytical commands are as follows:
### read count
readCounter --window 1000000 --quality 20 --chromosome 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22 sample.Raw.sort.dedup.bam > sample.cnv.wig
### Follow the ichorCNA installation procedure to install ichorCNA, using databases provided by the respective software
Rscript runIchorCNA.R --id {sample} --WIG {sample}.cnv.wig --ploidy "c(2)" --normal "c(0.95, 0.99, 0.995, 0.999)" --maxCN 3 --gcWig gc_hg19_1000kb.wig --mapWig map_hg19_1000kb.wig --centromere GRCh37.p13_centromere_UCSC-gapTable.txt --normalPanel HD_ULP_PoN_1Mb_median_normAutosome_mapScoreFiltered_median.rds --includeHOMD False --chrs "c(1:22)" --chrTrain "c(1:22)" --estimateNormal True --estimatePloidy True --estimateScPrevalence FALSE --scStates "c()" --txnE 0.9999 --txnStrength 10000 --outDir cnv --libdir ichorCNA_lib


## ARM-FSD

python frag_arm.py --bam {sample}.Raw.sort.dedup.bam --sample_name {sample} --outpath outdir --thread 16 --bedtools bin/bedtools-2.30.0 --database databases --reference hs39d5.fa
Rscript frag_wgs.r --sample {sample} --counts_by_width outdir/counts_by_width.csv --counts outdir/counts.csv --bingc outdir/bingc.csv --f_output outdir/{sample}.Raw.ARM-FSD.csv --range_start 100 --range_end 220 --step 5 --binsize 5 --ab databases/AB.rds

## MCMS
For MCMS feature analysis, you need to install: samtools, bedtools, bcftools,  ANNOVAR and picard.
The detailed analytical commands are as follows:
### mcms filter_bam
samtools view -h {sample}.Raw.sort.dedup.bam | awk '$9 != 0 {print $0}' > {sample}.simple.filtered.sam
### mcms gc
samtools view -h {sample}.simple.filtered.sam | awk -v a=300 -v b=-300 '(/^@/) || ($9 > b && $9 < a) {print $0}' | samtools view -h -bS - | bedtools-2.30.0 intersect -a stdin -b wgs_highconf_region.bed -sorted | bedtools-2.30.0 intersect -v -a stdin -b repeat_mask.sorted.bed -sorted | bedtools-2.30.0 intersect -v -a stdin -b windowmask.sorted.bed -sorted > {sample}.gc.rmask.short.bam && samtools index {sample}.gc.rmask.short.bam && java -XX:ParallelGCThreads=8 -Djava.io.tmpdir=./tmp -jar picard_2.9.4.jar CollectGcBiasMetrics  I={sample}.gc.rmask.short.bam R=hs37d5.fa WINDOW_SIZE=400 O={sample}.gc_bias_metrics.txt CHART={sample}.gc_bias_metrics.pdf S={sample}.gc_summary.txt
### mcms mut_reads
samtools view -H {sample}.simple.filtered.sam > {sample}.mut_only.paired.sam && samtools view {sample}.simple.filtered.sam | grep -Pe '^@|NM:i:1\t' | awk 'cnt[$1]++{if (cnt[$1]==2) print prev[$1]; print} {prev[$1]=$0}' >> {sample}.mut_only.paired.sam && samtools sort -@ 16 -T ./tmp {sample}.mut_only.paired.sam | samtools view -h -bS > {sample}.mut_only.paired.bam && samtools index {sample}.mut_only.paired.bam && bedtools-2.30.0 intersect -a {sample}.mut_only.paired.bam -b wgs_highconf_region.bed -sorted | bedtools-2.30.0 intersect -v -a stdin -b repeat_mask.sorted.bed -sorted | bedtools-2.30.0 intersect -v -a stdin -b windowmask.sorted.bed -sorted > {sample}.mut_paired.rmask.bam && samtools index {sample}.mut_paired.rmask.bam && samtools view {sample}.mut_paired.rmask.bam | gzip > {sample}.mut_paired.rmask.sam.gz
### mcms mpileup
bcftools-1.9 mpileup --ignore-overlaps --ignore-RG -q 60 -Q 30 -a DP,AD,ADF,ADR,SP,INFO/AD,INFO/ADF,INFO/ADR -Ou -f hs37d5.fa -o {sample}.mpileup.tmp1 {sample}.mut_paired.rmask.bam && bcftools-1.9 call --keep-alts --multiallelic-caller -p 0 -o {sample}.mpileup.tmp2 {sample}.mpileup.tmp1 && awk '$5 != "."' {sample}.mpileup.tmp2 | grep -v "INDEL" > {sample}.vcf && awk -v a=$(samtools view -c {sample}.Raw.sort.dedup.bam) '{print $1, $2, $2, $4, $5, "comments:", $8, a}' {sample}.vcf > {sample}.mut_only.vcf
### mcms annovar
perl table_annovar.pl {sample}.mut_only.vcf annovar/humandb/ -buildver hg19 -out ./ -protocol refGene,cytoBand,avsnp151 -operation g,r,f -remove -nastring . --otherinfo && cat {sample}.hg19_multianno.txt | gzip - > {sample}.SNP_in.hg19_multianno.txt.gz && cat {sample}.hg19_multianno.txt | grep -v "rs" | gzip - > {sample}.SNP_out.hg19_multianno.txt.gz

### mcms summary snp
Rscript summary_context.R {sample}.SNP_in.hg19_multianno.txt.gz {sample}.SNP_in {sample}.mut_paired.rmask.sam.gz
### mcms summary_without_snp
Rscript summary_context.R {sample}.SNP_out.hg19_multianno.txt.gz {sample}.SNP_out {sample}.mut_paired.rmask.sam.gz
### mcms baseline
python mc_generate.py --sample {sample} --mutation {sample}.SNP_out.long.GC_normalized.rds --summary {sample}.qc.summary.tsv --baseline baseline.SNP_out.MGI-T7.csv --output {sample}.MC.SNP_out.csv --level Raw --colprefix MC.SNP_out && python mc_generate.py --sample {sample} --mutation {sample}.SNP_in.long.GC_normalized.rds --summary {sample}.qc.summary.tsv --baseline baseline.SNP_in.MGI-T7.csv --level Raw --output {sample}.MC.SNP_in.csv --colprefix MC.SNP_in
### mcms fit pre
python prep_signature_fit.py --context {sample}.MC.SNP_in.csv --f_output {sample}.MC.SNP_in.long.tsv
### mcms fit
Rscript signature_fit.R --sample {sample} --context {sample}.MC.SNP_in.long.tsv --reference COSMIC_v3.3.1_SBS_GRCh37.tsv --output {sample}.MS.SNP_in.csv
### mcms generate
python merge_mc_ms.py --f_mc {sample}.MC.SNP_out.csv --f_ms {sample}.MS.SNP_in.csv --f_mc_out {sample}.Raw.MC.tmp --f_mcms_out {sample}.Raw.MCMS.csv

## Model Training
#### H2OAUTOML
gsml H2oAutoML --feature cnv.csv --train_info Train.info.list --d_output automl/cnv --nthreads 20 --prefix cnv --max_models 200 --pred_info Valid.info.list
gsml H2oAutoML --feature ARM-FSD.csv --train_info Train.info.list --d_output automl/ARM-FSD --nthreads 20 --prefix ARM-FSD --max_models 200 --pred_info Valid.info.list
gsml H2oAutoML --feature MCMS.csv --train_info Train.info.list --d_output automl/MCMS --nthreads 20 --prefix MCMS --max_models 200 --pred_info Valid.info.list
#### stat basemodel
example: gsml ModelStat --f_model {cnv.basemodel} --model_name {mid} --dataset Train,Train.info.list --dataset Valid,Valid.info.list --d_output automl/cnv --skip_combine_score
#### stack model
gsml pipe_combine_best_model --d_model_list cnv,./automl/cnv --d_model_list ARM-FSD,./automl/ARM-FSD --d_model_list MCMS,./automl/MCMS --d_output ModelSelect/  --train_info Train,Train.info.list --pred_info Valid,Valid.info.list --feature_list AllFeatures.combined.csv --n_top_models 2,3,4,5,6,7,8,9,10




