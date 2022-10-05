import numpy as np

#Fuzzy Rank-based Ensemble:
def getScore(model, test_imgs):
  res = model.predict(test_imgs)
  return res 

def generateRank1(score,class_no):
  rank = np.zeros([class_no,1])
  scores = np.zeros([class_no,1])
  scores = score
  for i in range(class_no):
      rank[i] = 1 - np.exp(-((scores[i]-1)**2)/2.0)
  return rank

def generateRank2(score,class_no):
  rank = np.zeros([class_no,1])
  scores = np.zeros([class_no,1])
  scores = score
  for i in range(class_no):
      rank[i] = 1 - np.tanh(((scores[i]-1)**2)/2)
  return rank

def doFusion(res1, res2, res3, label, class_no):
  cnt = 0
  id = []
  for i in range(len(res1)):
      rank1 = generateRank1(res1[i],class_no)*generateRank2(res1[i],class_no)
      rank2 = generateRank1(res2[i],class_no)*generateRank2(res2[i],class_no)
      rank3 = generateRank1(res3[i],class_no)*generateRank2(res3[i],class_no)
      rankSum = rank1 + rank2 + rank3
      rankSum = np.array(rankSum)
      scoreSum = 1 - (res1[i] + res2[i] + res3[i])/3
      scoreSum = np.array(scoreSum)
      
      fusedScore = (rankSum.T)*scoreSum
      cls = np.argmin(rankSum)
      if cls<class_no and label[i][cls]== 1:
          cnt += 1
      id.append(cls)
  print(cnt/len(res1))
  return id
