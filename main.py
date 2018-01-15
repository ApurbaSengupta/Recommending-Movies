"""

Created on Tue Nov 28 15:23:26 2017

@authors: Apurba Sengupta, Dhruv Desai

"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

'''

Question 1.1
    
In this project, we are returning to the MovieLens dataset. For this project 
you may use a smaller dataset to help with runtime: http://grouplens.org/
datasets/movielens/1m/. The goal of this project is to implement a type of 
recommender system, in which we are asked to choose for example 20 movies 
to advertise to users of a website. We want to maximize the chance that a 
user of the site will \like" some of the the movies. Recall that each user 
rates the movies by a number between 0-5 (if a user has not rated a movie, 
we simply let the corresponding ratingvalue to be 0). In your python 
program, construct a suitable representation of the data set in a matrix 
form.

'''

start_time1 = time.time()

def createCharacteristicMatrix(filename):

	# convert the ratings data file to a pandas dataframe and and then convert the dataframe into the required numpy matrix and return the matrix
    data_frame = pd.read_csv(filename, sep="::", usecols = [0, 1, 2], names = ['userID', 'movieID', 'rating'], engine = 'python')
    data_mat = np.array(data_frame.pivot(index = 'movieID', columns = 'userID', values = 'rating'))
    data_mat_rev = np.nan_to_num(data_mat)
    return data_mat_rev 

print "\n\n Reading the data into the characteristic matrix... \n\n"

X = createCharacteristicMatrix('./ratings.dat')

# set number of movies and users as the dimensions of the characteristic matrix
n_movies, n_users = X.shape[0], X.shape[1]

end_time1 = time.time() - start_time1

print "\n\n Time taken to generate the characteristic matrix consisting of", X.shape[0], "movies (in rows) and", X.shape[1], "users (in columns) =", round(end_time1, 2), "seconds\n\n"

'''

Question 1.2

The first step is to define an objective function which assigns to each subset
of the movies a real value representing how much the users "1ike" that subset. 
One way to define such an objective function is as follows. Let us 
first introduce some notation. Let n,m to be the number of users and movies, 
respectively. We also let r(i,j) denote the rating that user i assigns to
movie j. Given any subset A of the movies, we define F(A) = 1/n[sum(i = 1 to n)
{max r(i,j) for all j in A}] . Prove that the objective function F is both 
monotone and submodular.

'''

print "\n\n Defining the utility function ... \n\n"

# define the utility function
def F(A):
    
    if len(A) > 0:
        
        X_j = X[list(A)]
    
        F = float(np.sum(np.amax(X_j, axis = 0)))/float(n_users)
    
    else:
        
        F = 0
        
    return F

'''

Question 1.3

In the next part, we will implement the greedy submodular maximization algo-
rithm described in class. Note that, due to monotonicity and submodularity, the
greedy algorithm guarantees a solution A such that F(A) >= (1 - 1/e)F(A*), 
where A* is the true optimal set. Implement the greedy algorithm for 
maximization of F over all the subsets of movies that have cardinality 
at most k. Plot the objective values of the greedy algorithm versus k for 
k = 10, 20, 30, 40, 50.

'''

print "\n\n Running the 'Greedy' Submodular Maximization Algorithm ... \n\n"

start_time3 = time.time()

# list of required cardinalities
K = [10, 20, 30, 40, 50]

# list to hold the objective value of the greedy algorithm
greedy_objective_value_list = []

# list to hold the recommended movies
greedy_recommended_movies = []

# list to hold times
time_list_greedy = []

# for each cardinality value
for k in K:
    
    # empty list of recommended movies 
    A_greedy = set([])
    
    inter_start_greedy = time.time()
    
    # loop over the cardinality value
    for i in range(k):
        
        # find the movie index that gives maximum coverage, i.e., maximum F(A)
        e_opt = np.argmax([F(A_greedy.union(set([e]))) - F(A_greedy) for e in range(n_movies)])
        
        # update the list of recommended movies
        A_greedy = A_greedy.union(set([e_opt]))
    
    # time taken to create list of recommended movies
    inter_end_greedy = time.time() - inter_start_greedy
    
    # add the recommended movies to the list of recommended movies
    greedy_recommended_movies.append(A_greedy)
    
    # append objective values to the list of objective values
    greedy_objective_value_list.append(F(A_greedy))
    
    # add the time to list of times
    time_list_greedy.append(round(inter_end_greedy, 5))

end_time3 = time.time() - start_time3

print "\n\n Time taken to implement the 'Greedy' Submodular Maximization Algorithm =", round(end_time3, 5), "seconds\n\n"

# plot number of movies recommended v/s greedy algorithm utility values     
plt.title("\n Number of Movies Recommended v/s 'Greedy' Algorithm Utility Values\n")    
plt.xlabel("Number of Movies Recommended 'k'")
plt.ylabel("'Greedy' Algorithm Utility Values")
plt.plot(K, greedy_objective_value_list, label = "'Greedy' Algorithm")
plt.legend()
plt.show()

'''

Question 1.4

One way to make the greedy algorithm faster is to use the so-called "lazy" 
version. Expand on your implementation for part 2, and implement the lazy 
greedy algorithm. You should get the same greedy solution as in the previous 
part but with a smaller runtime. Record the runtime (in seconds) of both the 
greedy algorithm and its lazy version for k = 10; 20; 30; 40; 50 and plot the 
values.

'''

print "\n\n Running the 'Lazy Greedy' Submodular Maximization Algorithm ... \n\n"

start_time4 = time.time()

# list to hold the objective value of the greedy algorithm
lazy_greedy_objective_value_list = []

# list to hold the recommended movies
lazy_recommended_movies = []

# list to hold times
time_list_lazy = []

# for each cardinality value
for k in K:
    
    # empty list of recommended movies 
    A_lazy_greedy = set([])
    
    # start timer for creating list of recommended movies 
    inter_start_lazy = time.time()
    
    # loop over the cardinality value
    for i in range(k):
        
        if i == 0:
            
            # list of marginals 
            marginal_values_list = [F(A_lazy_greedy.union(set([e]))) - F(A_lazy_greedy) for e in range(n_movies)]
            
            # find the movie index that gives maximum coverage, i.e., maximum F(A)
            e_opt = np.argmax(marginal_values_list)
        
            # update the list of recommended movies
            A_lazy_greedy = A_lazy_greedy.union(set([e_opt]))
            
            # sort the list of marginals and remove the first element
            marginal_values_list_sorted = sorted(marginal_values_list)[::-1][1:]
            
            # sort according to the index of movies 
            movie_index_sorted = list(np.argsort(marginal_values_list))[::-1][1:]
            
        else:
            
            # check if delta(e2|A_i) > delta(e3|A_i-1) 
            while (F(A_lazy_greedy.union(set([movie_index_sorted[0]]))) - F(A_lazy_greedy)) < marginal_values_list_sorted[1]:
                
                # update the list of marginals
                marginal_values_list[movie_index_sorted[0]] = F(A_lazy_greedy.union(set([movie_index_sorted[0]]))) - F(A_lazy_greedy)
                
                # sort the list of marginals
                marginal_values_list_sorted = sorted(marginal_values_list)[::-1]
                
                # sort according to the index of movies
                movie_index_sorted = list(np.argsort(marginal_values_list))[::-1]
            
            # update list of recommended movies
            A_lazy_greedy = A_lazy_greedy.union(set([movie_index_sorted[0]]))
            
            # sort the list of marginals and remove the first element
            marginal_values_list_sorted = sorted(marginal_values_list)[::-1][1:]
            
            # sort according to the index of movies
            movie_index_sorted = list(np.argsort(marginal_values_list))[::-1][1:]
    
    # time taken to create list of recommended movies
    inter_end_lazy = time.time() - inter_start_lazy
        
    # add the recommended movies to the list of recommended movies
    lazy_recommended_movies.append(A_lazy_greedy)
    
    # append objective values to the list of objective values
    lazy_greedy_objective_value_list.append(F(A_lazy_greedy))
    
    # add the time to list of times
    time_list_lazy.append(round(inter_end_lazy, 5))

end_time4 = time.time() - start_time4

print "\n\n Time taken to implement the 'Lazy Greedy' Submodular Maximization Algorithm =", round(end_time4, 5), "seconds\n\n"

plt.title("\n Number of Movies Recommended v/s Utility Function Values\n")    
plt.xlabel("Number of Movies Recommended 'k'")
plt.ylabel("Utility Function Values")
plt.plot(K, greedy_objective_value_list, label = "'Greedy' Algorithm", linestyle = '-.', color = 'r')
plt.plot(K, lazy_greedy_objective_value_list, label = "'Lazy Greedy' Algorithm", linestyle = '--', color = 'y')
plt.legend()
plt.show()

plt.title("\n Number of Movies Recommended v/s Time Taken (in seconds)\n")    
plt.xlabel("Number of Movies Recommended 'k'")
plt.ylabel("Time Taken (in seconds)")
plt.plot(K, time_list_greedy, label = "'Greedy' Algorithm")
plt.plot(K, time_list_lazy, label = "'Lazy Greedy' Algorithm")
plt.legend()
plt.show()

end_time = time.time() - start_time1

print "\n\n Time taken by program to run =", round(end_time, 5), "seconds\n\n" 
