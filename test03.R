options(max.print = 1000000)

lines_text <- paste(readLines("R\\data_labels2.txt"))

labels <- list()
for (i in 1:length(lines_text)){
  labels <- append(labels,base::substr(lines_text[i], start=1, stop=3))
}

lines<-scan("R\\lcv.txt")

count<-1
for (i in 1:n) {
  for (j in i:n) {
    if (j > i) {
      upper_triangular_matrix[j, i] <- lines[count]
      count <- count + 1
    }
  }
}


hc <- hclust(as.dist(upper_triangular_matrix), "ward.D")

plot(hc,hang=-1,cex=0.5,labels=labels)
