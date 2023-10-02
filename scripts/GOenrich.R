#### RUN WITH: /usr/lib/R/bin/Rscript

library(missMethyl)
library(IlluminaHumanMethylation450kanno.ilmn12.hg19)

args = commandArgs(trailingOnly=TRUE)

cpglist <- read.delim(args[1], header=FALSE)
cpglist <- cpglist$V1


GOAssociation <- function(cpglist){
  # Get annotation
  ann <- getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)
  # Run gometh()
  gst <- gometh(sig.cpg = cpglist, 
                collection = c("GO", "KEGG"), 
                array.type = c("450K"),
                plot.bias = TRUE, 
                genomic.features = c("ALL"),
                anno = ann,
                prior.prob = TRUE) # TRUE - if FALSE a hypergeometric test will be conducted   


  return(gst)
}

keggAssociation <- function(cpglist){
  # Get annotation
  ann <- getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)
  # Run gometh()
  kegg <- gometh(sig.cpg = cpglist, 
                collection = c("KEGG"), 
                array.type = c("450K"),
                plot.bias = TRUE, 
                genomic.features = c("ALL"),
                anno = ann,
                prior.prob = TRUE) # TRUE - if FALSE a hypergeometric test will be conducted  

  return(kegg)
}     

goa = GOAssociation(cpglist)
kegg = keggAssociation(cpglist)

### Write results
PATH_out = strsplit(args[1], split = ".txt")[[1]]
dir.create(file.path(PATH_out))
write.csv(goa, paste(PATH_out, "GO_association.csv", sep="/"), row.names = FALSE)
write.csv(kegg, paste(PATH_out, "KEGG_association.csv", sep="/"), row.names = FALSE)
