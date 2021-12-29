# Importing the needed liberaries

import pandas as pd 
import numpy 
import random
import matplotlib.pyplot as plt
pd.options.display.max_columns = None
numpy.random.seed(0)

##################################################################################################################################
def get_patient_info():
    '''
    This function used for getting the information from the patient.
    
    Returns
    -------
    name            : Patient name 
    AgeCategory     : Patient age category (child, adult)
    Condition       : Condition Type (S for Stroke, SC for Spinal cord, and B for Brain injuries)
    elbow_typesNo   : The number of exercise types performing for the elbow.
    upperArm_typesNo: The number of exercise types performing for the upper arm.
    knee_typesNo    : The number of exercise types performing for the knee.
    wrist_typesNo   : The number of exercise types performing for the wrist.
    total_exeNo     : total number of exercies
    
    '''
    name = input("Welcome to Smart Rehabilitation!\nEnter your name please (◒‿◒): ") 
    #---------------------------
    AgeCategory = input("Hi " +name+ " Please enter your Age Category (A for Adult and C for Child):\n").lower()
    while AgeCategory not in ['a','c']:
        # Ask to enter an input agin
        AgeCategory = input('Unvalid letter,please enter A for Adult and C for Child). \n').lower()

    #---------------------------   
    Condition = input("Please enter your Condition Type (S for Stroke, SC for Spinal cord, and B for Brain injuries):\n").lower()
    while Condition not in ['s','sc','b']:
        # Ask to enter an input agin
        Condition = input("Unvalid letter,please enter S for Stroke, SC for Spinal cord, and B for Brain injuries:\n").lower()
    #---------------------------   
    
    elbow_typesNo = int(input("Please enter the number of exercises you prefer to perform for the elbow (1 for one type, and 2 for two types of exercises):\n"))
    while elbow_typesNo not in [1,2]:
        # Ask to enter an input agin
        elbow_typesNo = int(input("Unvalid number, please enter 1 for one type, and 2 for two types of exercises:\n"))
    #---------------------------   
    
    upperArm_typesNo  = int(input("Please enter the number of exercises you prefer to perform for the upper arm (1 for one type, and 2 for two types of exercises):\n")) 
    while upperArm_typesNo not in [1,2]:
        # Ask to enter an input agin
        upperArm_typesNo  = int(input("Unvalid number, please enter 1 for one type, and 2 for two types of exercises. \n")) 
    #---------------------------   
    
    knee_typesNo = int(input("Please enter the number of exercises you prefer to perform for the knee/lower leg (1 for one type, and 2 for two types of exercises):l\n"))
    while (knee_typesNo not in [1,2]):
        # Ask to enter an input agin
        knee_typesNo = int(input("Unvalid number, please enter 1 for one type, and 2 for two types of exercises):\n"))
    #---------------------------   
    
    wrist_typesNo = int(input("Please enter the number of exercises you prefer to perform for the wrist (0 for no exercise, and 1 for one type of exercise):\n"))
    while wrist_typesNo not in [0,1]:
        print("Unvalid number")
        # Ask to enter an input agin
        wrist_typesNo = int(input("Unvalid number, please enter 0 for no exercise, and 1 for one type of exercise:\n"))
    #---------------------------
    
    print("Preparing your optimal rehabilitation plan (◒‿◒). . .\n")
    total_exeNo = elbow_typesNo + upperArm_typesNo + knee_typesNo  + wrist_typesNo 
    print("Your rehabilitation plan is ready! Your plan is presented below with " + str(total_exeNo) +" exercises per day. ")
    
    return name, AgeCategory, Condition, elbow_typesNo, upperArm_typesNo, knee_typesNo, wrist_typesNo, total_exeNo
    
##################################################################################################################################

def create_general_dataset():
    '''
    This function used for creating the general dataset for each part in the body 
    
    Returns
    -------
    df : General dataset which contains all the details about the exercises.
    
    '''

    # Intialize dictionary of lists.
    Elbow_Data        = { 'Body':['Elbow', 'Elbow', 'Elbow', 'Elbow' , 'Elbow', 'Elbow'  , 'Elbow', 'Elbow' , 'Elbow', 'Elbow' , 'Elbow' ,'Elbow' ],
                         
                          'Exercises':['Elbow extensor using free weights ', 'Crawling', 'Elbow flexor using free weights ', 'Bear walking' , 
                                       'Elbow extensor using theraband  ', 'Lifting in parallel bars '  , 'Elbow Flexor using theraband ', 
                                       'Lifting in sitting using scale ' , 'Rotating forearm ', 'Forearm supination' ,
                                       'Learning forwards in a large ball ','Wheelbarrow walking on hands' ], 
                          
                          'Condition':['Stroke' , 'Stroke'  , 'Stroke'   , 'Stroke'  , 'Spinal cord injuries' , 'Spinal cord injuries',  
                                       'Spinal cord injuries' , 'Spinal cord injuries' , 'Brain injury ' , 'Brain injury' , 'Brain injury' , 'Brain injury '  ] , 
                          
                          'AgeCategory':['Adult', 'Child','Adult','Child','Adult','Child','Adult','Child','Adult','Child','Adult','Child']
                          } 
    
    
    UpperArm_Data     = {'Body':['Upper Arm', 'Upper Arm', 'Upper Arm', 'Upper Arm' , 'Upper Arm', 'Upper Arm'  , 'Upper Arm', 'Upper Arm' ,
                                 'Upper Arm', 'Upper Arm' , 'Upper Arm' ,'Upper Arm' ],
                         
                         'Exercises':['Shoulder external rotator using free weights ', 'Weight-bearing through one shoulder',
                                      'Shoulder extensor ', 'Crawling ' , 'Shoulder abductor using theraband ', 'Shoulder abductor Stritch in setting' ,
                                      'Shoulder adductor using theraband '  , 'Boxing in setting ', 'Push-ups in prone' , 'Reaching in four points kneeling',
                                      'Lowering and pushing-up in long setting ' , 'Crab-walking ' ], 
                         
                         'Condition':['Stroke' , 'Stroke'  , 'Stroke'   , 'Stroke'  , 'Spinal cord injuries' , 'Spinal cord injuries',  'Spinal cord injuries' ,
                                      'Spinal cord injuries' , 'Brain injury ' , 'Brain injury' , 'Brain injury' , 'Brain injury '  ] , 
                         
                         'AgeCategory':['Adult', 'Child','Adult','Child','Adult','Child','Adult','Child','Adult','Child','Adult','Child']
                         }
    
    
    KneeLowerleg_Data = {'Body':['Knee/ Lower leg', 'Knee/ Lower leg', 'Knee/ Lower leg', 'Knee/ Lower leg' , 'Knee/ Lower leg', 'Knee/ Lower leg'  ,
                                 'Knee/ Lower leg', 'Knee/ Lower leg' , 'Knee/ Lower leg', 'Knee/ Lower leg' , 'Knee/ Lower leg' ,'Knee/ Lower leg' ], 
                         
                         'Exercises':['Knee flexor using a device ', 'Squatting against a wall', 'Knee extensor using a device ', 'Seated walking' ,
                                      'Bending the knee in standing', 'Walking on heels' , 'Standing and setting  '  , 'Walking on tiptoes  ',
                                      'Hamster stretch in setting ' , 'Stepping up onto a block ' ,'Leg stretching using sandbag ','Stepping sideways onto a block '],
                         
    		             'Condition':['Stroke' , 'Stroke'  , 'Stroke'   , 'Stroke'  , 'Spinal cord injuries' , 'Spinal cord injuries', 
                                'Spinal cord injuries' , 'Spinal cord injuries' , 'Brain injury ' , 'Brain injury' , 'Brain injury' , 'Brain injury '  ] , 
                         
                         'AgeCategory':['Adult', 'Child','Adult','Child','Adult','Child','Adult','Child','Adult','Child','Adult','Child']
                         }
    
    
    Wrist_Data        = {'Body':['Wrist', 'Wrist', 'Wrist', 'Wrist' , 'Wrist', 'Wrist'  , 'Wrist', 'Wrist' , 'Wrist', 'Wrist' , 'Wrist' ],
                          'Exercises':['Wrist extensor using free weights', 'Finger extensor strengthening using an elastic band',
                                       'Wrist flexor using free weights','Finger Flexor using grip device','Pincer grip ','Propping in-side setting',
                                       'Wrist extensor using theraband','Wrist extensor using electrical simulation','Reaching in standing ',
                                       'Popping bubble wrap','Walking hands on a ball'] , 
                          
                          'Condition':['Stroke' , 'Stroke'  , 'Stroke'   , 'Stroke'  , 'Spinal cord injuries' , 'Spinal cord injuries',
                                       'Spinal cord injuries' , 'Brain injury ' , 'Brain injury' , 'Brain injury' , 'Brain injury '  ] ,
                          
                          'AgeCategory':['Adult', 'Child','Adult','Child','Adult','Child','Adult','Adult','Child','Adult','Child'] 
                          } 
    
    # Convert Dictionaries to DataFrame using Pandas
    Elbow_df        = pd.DataFrame(Elbow_Data) 
    UpperArm_df     = pd.DataFrame(UpperArm_Data) 
    KneeLowerleg_df = pd.DataFrame(KneeLowerleg_Data) 
    Wrist_df        = pd.DataFrame(Wrist_Data) 
    
    # Concate All DataFrames
    # Stack the DataFrames on top of each other
    df = pd.concat([Elbow_df, UpperArm_df , KneeLowerleg_df , Wrist_df ], axis=0)
    df = df.reset_index(drop=True)
    pd.set_option('display.width' , 10000)
    
    return df
##################################################################################################################################
def customized_dataset(AgeCategory, Condition,df):
    
    '''
    This function return the a new dataset based on age condition type .
    
    Parameters
    ----------
    AgeCategory : Number of solution per population
    Condition   : Number of weights used
    df          : General dataset which contains all the details about the exercises.
    
    Returns
    -------
    return the 3 parameters of the equation

    new_df      : The exercises dataset customized for the patient based on the entered information.
    age         : The patient's age equal 1 if the patient is adults and 2 if a child.
    cond        : The patient's condition equal 1 if for Stroke, 2 for Spinal cord injuries, and 3 for Brain injury.
    
    '''

    if AgeCategory == 'a' :
         new_df = df[(df.AgeCategory == 'Adult')]
         age = 1
    
         
    elif AgeCategory == 'c'  :
         new_df = df[(df.AgeCategory == 'Child')]
         age = 2
         
         
    if  Condition == 's' :
         new_df = new_df[(new_df.Condition == 'Stroke')]
         cond = 1
    
        
    elif Condition == 'sc' : 
         new_df = new_df[(new_df.Condition == 'Spinal cord injuries')]
         cond = 2
    
        
    elif Condition == 'b'   : 
         new_df = new_df[(new_df.Condition == 'Brain injury ')]
         cond = 3
         
    return new_df, age, cond

##################################################################################################################################
def cal_pop_fitness(equation_inputs, population):
    '''
    This function calculating the fitness value of each solution in the current population
    by calculating the sum of products between each input and its corresponding weight.
    
    Parameters
    ----------
    equation_inputs : List of the three parameters.
    population      : The corresponding weights.

    Returns
    -------
    fitness         : The sum of products between each input and its corresponding weight.
    '''
    fitness = numpy.sum( population * equation_inputs, axis=1)
    
    return fitness
##################################################################################################################################
def get_probability_list(population_fitness):
    
    #fitness = population_fitness.values()
    total_fit = float(sum(fitness))
    relative_fitness = [f/total_fit for f in fitness]
    probabilities = [sum(relative_fitness[:i+1]) 
                     for i in range(len(relative_fitness))]
    return probabilities

##################################################################################################################################

def roulette_wheel_selection(population, fitness, n_parents):
    '''
    This function Selecting the best individuals in the current generation as parents 
    for producing the offspring of the next generation.
    
    Parameters
    ----------
    population : The corresponding weights of the population.
    fitness    : The sum of products between each input and its corresponding weight.
    n_parents : number of parents mating.
    
    Returns
    -------
    parents   : The sum of products between each input and its corresponding weight.
    '''
    
    probabilities = get_probability_list(fitness)
    
    chosen_parents = numpy.empty((n_parents, population.shape[1]))
    
    for p in range(n_parents):
        r = random.random()
        for (i, individual) in enumerate(population):
            if r <= probabilities[i]:
                chosen_parents[p, :] = population[i, :]
                #chosen_parents.append(list(individual))
                break
            
    return chosen_parents

##################################################################################################################################

def single_point_crossover(parents, offspring_size, rate):

    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually it is at the center.
    crossover_point = numpy.uint8(offspring_size[1]/2)
    
    if random.random() < rate:
        for k in range(offspring_size[0]):
             # Index of the first parent to mate.
             parent1_idx = k % parents.shape[0]
             # Index of the second parent to mate.
             parent2_idx = (k+1) % parents.shape[0]
             # The new offspring will have its first half of its genes taken from the first parent.
             offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
             # The new offspring will have its second half of its genes taken from the second parent.
             offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    
    return offspring

##################################################################################################################################
def uniform_mutation(offspring, rate):
    # Mutation changes a single gene in each offspring randomly.
    
    for i in range(offspring.shape[0]):
        # The random value to be added to the gene.
        val = numpy.random.uniform(-1.0, 1.0, 1)
        
        if random.random() < rate:
           offspring[i, 2] = offspring[i, 2] + val
           
    return offspring   
##################################################################################################################################
##################################################################################################################################


"""
The target is to maximize this equation:
    y = w1 * x1 + w2 * x2 + w3 * x3
    where (x1,x2,x3) is (Age Category, Condition, total number of exercies)
    we want to reach to the best values for the 3 weights w1, w2, w3
    We are going to use the genetic algorithm for the best possible values after a number of generations.
"""

_, AgeCategory, Condition, elbow_typesNo, upperArm_typesNo, knee_typesNo, wrist_typesNo, total_exeNo = get_patient_info()
general_df = create_general_dataset()
custom_df, age, cond = customized_dataset( AgeCategory, Condition, general_df)
 

# Inputs of the equation.
equation_param = [age,total_exeNo,cond]
# Genetic algorithm parameters
n_parents_mating = 4
solution_per_pop = 30
n_weights        = 3 
crossover_rate   = 0.8
mutation_rate    = 0.4
          
n_generation     = 100
iters = 20

# total population size will have solution_per_pop chromosome where each chromosome has n_weights genes.
pop_total_size = (solution_per_pop, n_weights) 

# weights initialization
new_population = numpy.random.uniform(low = 0, high = 1, size = (solution_per_pop, n_weights))


fitness_df = pd.DataFrame()
fits  = []

for i in range(iters):
    
    for g in range(n_generation):
        
        # Measuring the fitness of each chromosome in the population.
        fitness = cal_pop_fitness(equation_param, new_population)
       
        # Selecting the best parents in the population for mating.
        parents = roulette_wheel_selection(new_population, fitness,  n_parents_mating)
    
        # Generating next generation using crossover.
        offspring_crossover = single_point_crossover(parents,(pop_total_size[0]-parents.shape[0], n_weights), crossover_rate)
    
        # Adding some variations to the offsrping using mutation.
        offspring_mutation =  uniform_mutation(offspring_crossover, mutation_rate)
    
        # Creating the new population based on the parents and offspring.
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :]  = offspring_mutation
        
        # The best result in the current iteration.
        #print("Best result : ", numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))
    
        fits.append(numpy.max(numpy.sum(new_population * equation_param, axis=1)))
    
        fitness_df['fitness'] = pd.Series((numpy.sum(new_population * equation_param, axis=1)))

##################################################################################################################################

# Getting the best solution after iterating finishing all generations.

fitness = cal_pop_fitness(equation_param, new_population)
# Then return the index of that solution corresponding to the best fitness.
       
fitness_avg = numpy.array([sum(fits)/len(fits)])
print("\nThe Average Fitness{}\n".format(fitness_avg) )
#print("\nThe Best Fitness [{}]\n".format(numpy.max(fitness) ) )


fitness_df = fitness_df.reset_index(drop= True)
custom_df = custom_df.reset_index(drop= True)

df = pd.concat([custom_df , fitness_df] ,axis =1)
df = df.dropna()
df = df.sort_values(by="fitness" , ascending =False)
df = df.reset_index(drop= True)


Elbow = df[(df.Body == 'Elbow')]
Elbow = Elbow[:elbow_typesNo]
Elbow = list(Elbow['Exercises'])



UpperArm = df[(df.Body == 'Upper Arm')]
UpperArm = UpperArm[:upperArm_typesNo]
UpperArm = list(UpperArm['Exercises'])


Knee = df[(df.Body == 'Knee/ Lower leg')]
Knee = Knee[:knee_typesNo ]
Knee = list(Knee['Exercises'])


Wrist = df[(df.Body == 'Wrist')]
Wrist = Wrist[:wrist_typesNo]
Wrist = list(Wrist['Exercises'])


print("\nElbow : \n")
for i , elb in enumerate(Elbow):
    print(str(i+1) +". "+ str(elb)+ ".")
 
print("\nUpper Arm : \n")    
for i , arm in enumerate(UpperArm):
    print(str(i+1) +". "+ str(arm) + ".")
    
print("\nKnee/ Lower leg : \n")    
for (i , j) in enumerate(Knee):
    print(str(i+1) +". "+ str(j) + ".")

print("\nWrist : \n")    
for (i , j) in enumerate(Wrist):
    print(str(i+1) +". "+ str(j) + ".")
    
print('\nHope you fast recovery!\n')
    
gens = n_generation * iters +1
X = numpy.arange(1, gens)

plt.plot(X[0:gens:100] , fits [0:gens:100],marker='*')
plt.title('GA Performance')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.grid(True)
plt.show()
