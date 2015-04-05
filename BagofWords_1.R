###############################################################################
# Bag of Words
# author: khmelkoff
###############################################################################

# Загружаем библиотеки, готовим обучающую выборку #############################

library(tm)
library(SnowballC)
library(e1071)
library(caret)
library(randomForest)


training <- read.delim(unz("labeledTrainData.tsv.zip", 
                               "labeledTrainData.tsv"),
                               header = TRUE,
                               sep = "\t",
                               quote = "",
                               as.is=TRUE)

# Смотрим на данные ###########################################################
# Проверяем размерность, инспектируем первое ревю
dim(training)
r1_length <- nchar(as.character(training[1,3]))
r1 <- training[1,3] 
paste(substr(r1,1,700),"...")

# Избавляемся от HTML тегов
cleanHTML <- function(x) {
    return(gsub("<.*?>", "", x))
}
r1 <- cleanHTML(r1)

# Оставляем только текст, убираем однобуквенные слова и слова нулевой длины
onlyText <- function(x) {
    x <- gsub("'s", "", x) 
    return(gsub("[^a-zA-Z]", " ", x)) 
}
r1 <- onlyText(r1)

# Токенизируем
tokenize <- function(x) {
    x <- tolower(x)
    x <- unlist(strsplit(x, split=" "))
}

r1 <- tokenize(r1)
r1 <- r1[nchar(r1)>1]   

# Создаем список стоп-слов
stopWords <- stopwords("en")
r1 <- r1[!r1 %in% stopWords]
r1[1:20]

# Обрабатываем все 25000 записей
rws <- sapply(1:nrow(training), function(x){
    # Прогресс-индикатор
    if(x %% 1000 == 0) print(paste(x, "reviews processed")) 
    
    rw <- training[x,3]
    rw <- cleanHTML(rw)
    rw <- onlyText(rw)
    rw <- tokenize(rw)
    rw <- rw[nchar(rw)>1]
    rw <- rw[!rw %in% stopWords]
    
    paste(rw, collapse=" ") # Снова склеиваем в текст
})


# Строим "Мешок слов" #########################################################
train_vector <- VectorSource(rws) # Вектор
train_corpus <- Corpus(train_vector, # ?Корпус
                       readerControl = list(language = "en"))
train_bag <- DocumentTermMatrix(train_corpus, # Спец. матрица документы/термины
                                control=list(stemming=TRUE))

train_bag <- removeSparseTerms(train_bag, 0.9982) # Убираем слишком редкие термины
dim(train_bag)

# Смотрим на перечень наиболее распространенных терминов
hight_freq <- findFreqTerms(train_bag, 5000, Inf)
inspect(train_bag[1:4, hight_freq[1:10]])

# Из специальной матрицы формируем обучающий датафрейм

train_df <- data.frame(inspect(train_bag[1:25000,]))
train_df <- cbind(training$sentiment, train_df)

# Сокращенный датафрейм для статьи
# train_df <- data.frame(inspect(train_bag[1:1000,hight_freq]))
# train_df <- cbind(training$sentiment[1:1000], train_df)

names(train_df)[1] <- "sentiment"
vocab <- names(train_df)[-1] # Формируем словарь (для тестовой выборки)

# ?Убираем ненужное
rm(train_bag)
rm(train_corpus)
rm(train_vector)
rm(training)
rm(rws)

# Выращиваем Случайный лес ####################################################
t_start <- Sys.time()
set.seed(3113)
forest <- train(as.factor(sentiment) ~., data=train_df,
                method="rf",
                trControl=trainControl(method="cv",number=5),
                prox=TRUE,
                ntree=100,
                do.trace=10,
                allowParallel=TRUE)
t_end <- Sys.time()

# Смотрим на модель и на время обучения
t_end-t_start
print(forest)


# Загружаем и обрабатываем контрольную выборку ################################
testing <- read.delim(unz("testData.tsv.zip", 
                               "testData.tsv"),
                               header = TRUE,
                               sep = "\t",
                               quote = "")

# Проверяем размерность
dim(testing)

# Обрабатываем тестовые ревю
rws <- sapply(1:nrow(testing), function(x){
    
    if(x %% 1000 == 0) print(paste(x, "reviews processed")) 
    
    rw <- testing[x,2]
    rw <- cleanHTML(rw)
    rw <- onlyText(rw)
    rw <- tokenize(rw)
    rw <- rw[nchar(rw)>1]
    rw <- rw[!rw %in% stopWords]
    
    paste(rw, collapse=" ")
})

# Формируем вектор ревю, строим корпус, формируем матрицу документы/термины ###
test_vector <- VectorSource(rws)
test_corpus <- Corpus(test_vector, 
                       readerControl = list(language = "en"))

test_bag <- DocumentTermMatrix(test_corpus,
                               control=list(stemming=TRUE,
                                dictionary = vocab))

test_df <- data.frame(inspect(test_bag[1:25000,]))
sentiment <- rep(0, 25000)
test_df <- cbind(testing[1:25000,1], sentiment, test_df)
names(test_df)[1] <- "id"

# Прогнозируем сантимент ######################################################

test_df[,2] <- predict(forest, newdata = test_df)

# Сохраняем результата в csv
write.csv(test_df[,1:2], file="Submission.csv", 
          quote=FALSE,
          row.names=FALSE)

