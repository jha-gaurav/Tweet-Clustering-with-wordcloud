#Data Science Certification Project
#Project: NFL Dataste Analysis (contains tweets)

#include the required libraries
library(NLP)
library(tm)
library(slam)
library(dplyr)
library(animation)
library(wordcloud)

#Step 0: Set the working directory to the location of the dataset,
#        whch in this case is the location of source file.

#Step 1: Read the dataset into a dataframe

cert_df <- read.csv("NFL_SocialMedia_sample_data1.csv", header = TRUE, sep = ",",
                    stringsAsFactors = FALSE)
#inspect the structure and first few records of the dataset
str(cert_df)
head(cert_df)

#All seems good until this point. Proceed to next step.
#Step 2: Separate out the log and time columns

cert_df_comb <- cert_df[, c(1,3)]

#Inspect the objects created
cert_df_comb


#Now move to Step 3
#Create the corpus of the content
cert_corp_step3 <- Corpus(VectorSource(cert_df_comb$content))

#Inspect the corpus created
print(cert_corp_step3)
inspect(cert_corp_step3 [1:3])

#Now Step 4
#Convert the corpus into lower case

#cert_corp_step4 <- tm_map(cert_corp_step3, content_transformer(tolower))
cert_corp_step4 <- tm_map(cert_corp_step3, tolower)

#Inspect the corpus
inspect(cert_corp_step4)

#Step 5: emove the english stopwords

cert_corp_step5 <- tm_map(cert_corp_step4, removeWords, stopwords())

#Inspect the corpus
inspect(cert_corp_step5)

#Step 6: Remove punctuations
cert_corp_step6 <- tm_map(cert_corp_step5, removePunctuation)

#Inspect the corpus
inspect(cert_corp_step6)

#Step 7: Remove numbers
cert_corp_step7 <- tm_map(cert_corp_step6, removeNumbers)

#Inspect the corpus
inspect(cert_corp_step7)

#Step 8: Eliminate the white spaces
cert_corp_step8 <- tm_map(cert_corp_step7, stripWhitespace)
cert_corp_step8 <- tm_map(cert_corp_step8, PlainTextDocument)

#Inspect the corpus
inspect(cert_corp_step8)

#Step 9: Create a Document term matrix
cert_dtm <- DocumentTermMatrix(cert_corp_step8)

#Inspect the matrix
cert_dtm

#Step 10: Determine the term-frequency and tf-idf

cert_dtm_wtmat <- weightTfIdf(cert_dtm, normalize = TRUE)

#Convert the weighted tf-idf matrix to matrix
cert_matrix <- as.matrix(cert_dtm_wtmat)
rownames(cert_matrix) <- 1:nrow(cert_matrix)

#Normaize the matrix created
#Create a function normalize
normalize <- function(x) { return ((sum (x ^ 2)) ^ 0.5)}

#apply the normalize function to the matrix
cert_mat_norm <- apply(cert_matrix, MARGIN = 1, FUN=normalize)


#Apply Kmeans clustering
set.seed(1234)
cert_clust <- kmeans(cert_mat_norm, 10, iter.max = 8)
cert_clust


#write output file
cert_out <- cbind(as.character(cert_df$content), cert_clust$cluster)
write.csv(cert_out, "Cluster_Out.csv")


#Step 15: Find top 5 words from each cluster and write a file âTopWords.csvâ
#a. Column-bind the created DocumentTermMatrix with the cluster number
#b. Create subset of the cert_dtm cluster-wise
#c. Remove the cluster number column from all the subsets
#d. Now we have all the clusters in separate DTMs. Run findFreqTerms() to
#   find the frequently occuring terms.

cert_dtm_mat <- as.matrix(cert_dtm)
cert_dtm_mat <- cbind(cert_dtm_mat, cert_clust$cluster)
rownames(cert_dtm_mat) <- 1:nrow(cert_dtm_mat)
a <- as.data.frame(cert_dtm_mat, stringsAsFactors = FALSE)

#for wordcloud

#a1_final <- NULL
for (k in 1:10){
#Create a subset for each cluster
  a1 <- subset(a[,1:ncol(a) - 1], a[,ncol(a)] == k)
  a_colname <- colnames(a1)
  a1_colsum <- colSums(a1)
  a1_merge <- data.frame(cbind(a_colname, a1_colsum), stringsAsFactors = FALSE)
  a1_index <- as.data.frame(a1_merge[order(a1_merge$a1_colsum, decreasing = TRUE),], stringsAsFactors = FALSE)
#Create wordcloud for each cluster  
  mypath <- file.path(paste("Cluster_", k, ".jpg", sep = ""))
  jpeg(mypath)
  wordcloud(a1_merge$a_colname, min.freq = 5, max.words = 40, scale = c(3, 1), random.order = FALSE)
  dev.off()  
#Finished creating the wordcloud for each cluster
#Now prepare to write the file for TopWords in each cluster
  a1_write <- cbind(k,
                  nrow(a1),
                  as.character(a1_index$a_colname[1:5]),
                  a1_index$a1_colsum[1:5],
                  c(1:5))
  rownames(a1_write) <- as.character(a1_index$a_colname[1:nrow(a1_write)])
  if (k == 1)
  {
    a1_final <- a1_write
  }
  else
  {
    a1_final <- rbind(a1_final, a1_write)
  }
}
colnames(a1_final) <- c("Log Group", "Log Count", "Top Words", "Word Count", "Counter")
write.csv(a1_final, "TopWords.csv")
