## NOTE: python
# ans = ans.split(',')
# pickle.dump(ans, open("../ans.pkl", 'wb'))

# Unique answers
ans <- rbind(paste(train$answerA), paste(train$answerB), paste(train$answerC), paste(train$answerD))
ans <- as.list(ans)
ans <- unique(ans)
answer <- NULL
for(i in 1:length(ans)){
  answer <- paste(answer, ans[i], sep=',')
}
write.table(answer, "../answer.txt", sep=",")


# Correct answers
correct <- NULL
for(i in 1:length(train$correctAnswer)){
  choice <- paste('answer', train$correctAnswer[i], sep='')
  correct <- paste(correct, train[i, choice], sep=', ')
}
write.table(correct, "../correct.csv", sep='\n')
