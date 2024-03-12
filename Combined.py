
# In[1]:


import numpy as np
import random


# In[2]:


File_data = np.loadtxt("input1.txt", dtype=int) 

X = [0]*File_data.shape[1]

for i in range(File_data.shape[1]):
	X[i] = File_data[:,i]

X = np.array(X)
print(X)

File_data1 = np.loadtxt("input2.txt", dtype=float) 

users = File_data1


# In[3]:


n_of_docs = X.shape[0]
n_of_shingles = X.shape[1]

rand = [1,2,3,4,5,6]
random.seed(1)

permutation_matrix = [0]*4

for i in range(n_of_docs):
	rand = [1,2,3,4,5,6]
	random.shuffle(rand)
	permutation_matrix[i] = rand.copy()


# In[4]:


permutation_matrix = np.array(permutation_matrix)


# In[5]:


# Part 1

signature_matrix = np.zeros((n_of_docs, n_of_docs), dtype=int)

for i in range(permutation_matrix.shape[0]):
	for j in range(X.shape[0]):
		temp = permutation_matrix[i] * X[j]
		idx = np.where(temp != 0)[0]
		signature_matrix[i][j] = min(temp[idx])
		
		
print("Permutation Matrix:")
print(permutation_matrix)
print("\nSignature Matrix:")
print(signature_matrix)


# In[6]:

file = open('./output.txt', 'w')

for i in range(signature_matrix.shape[0]):
	for j in range(signature_matrix.shape[1]):
		file.write(str(signature_matrix[i][j]))
		file.write(" ")
	file.write("\n")
	
for i in range(X.shape[0]):
	for j in range(i+1,X.shape[0]):
		temp = X[i] + X[j]
		n1 = np.sum(temp == 1)
		n2 = np.sum(temp == 2)
		col = float(n2/(n1+n2))
		print("col (d{}, d{}):".format(i+1,j+1), col)
		file.write("{:.2f} ".format(col))
file.write("\n")
		
for i in range(signature_matrix.shape[0]):
	for j in range(i+1,signature_matrix.shape[0]):
		n = np.sum(signature_matrix[:,i] == signature_matrix[:,j])
		sig = float(n/signature_matrix.shape[0])
		print("sim (d{}, d{}):".format(i+1,j+1), sig)
		file.write("{:.2f} ".format(sig))
file.write("\n")

probability = 0.6
bands = 10
rows = 4
simi = 1-(1-probability**rows)**bands 
print(simi)
file.write("{:.2f} ".format(simi))
file.write("\n")


# In[ ]:




# Collaborative



# In[2]:


idx = np.where(users == -1)
users[idx] = 0
new_users = users[-2:]
users = np.delete(users, -2, axis=0)
users = np.delete(users, -1, axis=0)


# In[3]:


print(*users)


# In[4]:


users.shape


# In[5]:


def cosine_similarity(user1, user2):
	dot_prod = np.dot(user1, user2)
	mag_u1 = np.linalg.norm(user1)
	mag_u2 = np.linalg.norm(user2)
	return round(dot_prod/(mag_u1*mag_u2), 2)


# In[6]:


cos_new_users = cosine_similarity(new_users[0], new_users[1])
cos_new_users


# In[7]:


cos_new_user1 = [0]*20
cos_new_user2 = [0]*20
for i in range(len(new_users)):
	for j in range(len(users)):
		if(i==0):
			cos_new_user1[j] = cosine_similarity(new_users[i],users[j])
			file.write(str(cos_new_user1[j]))
			file.write(" ")
		else:
			cos_new_user2[j] = cosine_similarity(new_users[i],users[j])
			file.write(str(cos_new_user2[j]))
			file.write(" ")
	file.write("\n")


# In[8]:


print("Cosine Similarity User1")
print(*cos_new_user1)
print()


# In[9]:


print("Cosine Similarity User2")
print(*cos_new_user2)
print()


# In[10]:


def normalize_utility_matrix(user):
	tot = np.sum(user)
	num = np.sum(user != 0)
	avg = tot/num
	temp = np.zeros(user.shape[0])
	for i in range(user.shape[0]):
		if(user[i]!=0):
			temp[i] = user[i] - avg
		else:
			temp[i] = user[i]
	
	return temp


# In[11]:


print("Utility Matrix usgin mean centering")

norm_users = users.copy()
for i in range(users.shape[0]):
	norm_users[i] = normalize_utility_matrix(users[i])
	print(*norm_users[i])
	for val in norm_users[i]:
		file.write(str(round(val, 2)))
		file.write(" ")
	file.write("\n")
	
	
	
norm_new_users = new_users.copy()
for i in range(new_users.shape[0]):
	norm_new_users[i] = normalize_utility_matrix(new_users[i])
	print(*norm_new_users[i])
	for val in norm_new_users[i]:
		file.write(str(round(val, 2)))
		file.write(" ")
	file.write("\n")


# In[12]:


# # def mean_exclude_zeros(user):
# #     tot = np.sum(user)
# #     num = np.sum(user != 0)
# #     avg = tot/num
# #     return avg
	
# def pearson_corr_coef(user1, user2):
#     num = 0
#     avg1 = 0
#     avg2 = 0
#     user1_shared = []
#     user2_shared = []
#     for i in range(len(user1)):
#         if(user1[i]!=0 and user2[i]!=0):
#             user1_shared.insert(0,user1[i])
#             user2_shared.insert(0,user2[i])
#             avg1+=user1[i]
#             avg2+=user2[i]
#             num+=1
			
#     avg1 = avg1/num
#     avg2 = avg2/num
# #     user1 = np.array(user1_shared)
# #     user2 = np.array(user2_shared)
	
#     print(avg1, user1_shared)
#     print(avg2, user2_shared)
	
#     cov = 0
#     for i in range(len(user2_shared)):
#         cov += (user1_shared[i] - avg1)*(user2_shared[i] - avg2)
	
#     std1 = 0
#     for i in range(user1.shape[0]):
#         std1 += (user1[i] - avg1)**2
		
#     std2 = 0
#     for i in range(user1.shape[0]):
#         std2 += (user2[i] - avg2)**2
		
#     std1 = np.sqrt(std1)
#     std2 = np.sqrt(std2)
#     print(cov, std1, std2, std1*std2)
	
#     return round(cov/(std1*std2),2)


# In[13]:


# corr_new_user1 = [0]*20
# corr_new_user2 = [0]*20
# for i in range(len(new_users)):
#     for j in range(len(users)):
#         if(i==0):
#             corr_new_user1[j] = pearson_corr_coef(new_users[i],users[j])
#         else:
#             corr_new_user2[j] = pearson_corr_coef(new_users[i],users[j])


# In[14]:


corr_new_user1 = [0]*20
corr_new_user2 = [0]*20
for i in range(len(norm_new_users)):
	for j in range(len(norm_users)):
		if(i==0):
			corr_new_user1[j] = cosine_similarity(norm_new_users[i],norm_users[j])
			file.write(str(corr_new_user1[j]))
			file.write(" ")
		else:
			corr_new_user2[j] = cosine_similarity(norm_new_users[i],norm_users[j])
			file.write(str(corr_new_user2[j]))
			file.write(" ")
	file.write("\n")


# In[15]:


print("Pearson correlation coefficient - User1")
print(*corr_new_user1)
print()


# In[16]:


print("Pearson correlation coefficient - User2")
print(*corr_new_user2)
print()


# In[17]:


k = 3

corr_new_user1_copy = corr_new_user1.copy()
corr_new_user1_copy.sort(reverse=True)

corr_new_user2_copy = corr_new_user2.copy()
corr_new_user2_copy.sort(reverse=True)


# In[18]:


print("Top 3 highest correlation - User1")
print(*corr_new_user1_copy[:k])
print()


# In[19]:


print("Top 3 highest correlation - User2")
print(*corr_new_user2_copy[:k])
print()


# In[20]:


near_neigh_user1 = [0]*k
ni = 0
for i in corr_new_user1_copy[:k]:
	near_neigh_user1[ni] = np.where(corr_new_user1 == i)[0][0]
	ni+=1
	
print("Top 3 highest correlation users for new user1")
print(*near_neigh_user1[:k])
print()


# In[21]:


near_neigh_user2 = [0]*k
ni = 0
for i in corr_new_user2_copy[:k]:
	near_neigh_user2[ni] = np.where(corr_new_user2 == i)[0][0]
	ni+=1
	
print("Top 3 highest correlation users for new user2")
print(*near_neigh_user2[:k])
print()


# In[22]:


def weighted_average(k, new_user, users, near_neigh_user, corr_new_user_results):
	missing_rating_index = np.where(new_user == 0)[0]
	similar_user_rating = np.zeros([len(missing_rating_index), k])
	
	for i in range(len(missing_rating_index)):
		for j in range(k):
			similar_user_rating[i][j] = users[near_neigh_user[j]][missing_rating_index[i]]
	
	for i in range(len(missing_rating_index)):
		numero = np.sum(corr_new_user_results*similar_user_rating[i])
		denom = np.sum(corr_new_user_results)
		new_user[missing_rating_index[i]] = round(numero/denom,2)
	
	return new_user


# In[23]:


print("Filled all missing ratings for new user1")
print(*weighted_average(k, new_users[0].copy(), users, near_neigh_user1, corr_new_user1_copy[:k]))
print()
a = new_users.copy()
a[0] = weighted_average(k, new_users[0].copy(), users, near_neigh_user1, corr_new_user1_copy[:k])

for val in a[0]:
		file.write(str(round(val, 2)))
		file.write(" ")
file.write("\n")


# In[24]:


print("Filled all missing ratings for new user2")
print(*weighted_average(k, new_users[1].copy(), users, near_neigh_user2, corr_new_user2_copy[:k]))
a[1] = weighted_average(k, new_users[1].copy(), users, near_neigh_user2, corr_new_user2_copy[:k])
for val in a[1]:
		file.write(str(round(val, 2)))
		file.write(" ")
file.write("\n")
print()


# In[ ]:
    
users = np.concatenate((users, new_users))
# utility_matrix = users
item_matrix = users.T
overall_avg = np.mean(users[users != 0])
user_biases = np.nanmean(users - overall_avg, axis=1)
item_biases = np.nanmean(users - overall_avg - np.expand_dims(user_biases, axis=1), axis=0)
baseline_predictions = overall_avg + np.expand_dims(user_biases, axis=1) + item_biases
deviations = users - baseline_predictions
item_similarity = np.dot(deviations.T, deviations)
item_norms = np.linalg.norm(deviations, axis=0)
item_similarity /= np.outer(item_norms, item_norms)
k = 3
recommendations = []
for item_idx in range(item_similarity.shape[0]):
    neighbors = np.argsort(item_similarity[item_idx])[:-k-1:-1]
    neighbor_ratings = deviations.T[neighbors]
    predicted_rating = baseline_predictions[:, item_idx] + np.sum(neighbor_ratings, axis=0) / k
    recommendations.append(predicted_rating)

recommendations = np.array(recommendations).T
updated_users = np.where(users == 0, recommendations, users)
updated_users_rounded = np.round(updated_users, 2)

print("Updated Utility Matrix (Rounded to 2 decimal places):\n", updated_users_rounded[-2:])

for val in updated_users_rounded[-2]:
		file.write(str(round(val, 2)))
		file.write(" ")
file.write("\n")


# In[24]:

for val in updated_users_rounded[-1]:
		file.write(str(round(val, 2)))
		file.write(" ")
file.write("\n")

file.flush()
print("flush")