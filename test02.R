options(max.print = 1000000)

lines<-scan("R\\lcv.txt")

# リストの要素を上三角行列に設定
count<-1
for (i in 1:n) {
  for (j in i:n) {
    if (j > i) {
      upper_triangular_matrix[j, i] <- lines[count]
      count <- count + 1
    }
  }
}

labels <- list()
for (i in 1:24){
  labels <- append(labels, 'apo')
}

for (i in 1:24){
  labels <- append(labels, 'ben')
}

for (i in 1:24){
  labels <- append(labels, 'try')
}

hc <- hclust(as.dist(upper_triangular_matrix), "ward.D")

plot(hc,hang=-1,cex=0.5,labels=labels)

